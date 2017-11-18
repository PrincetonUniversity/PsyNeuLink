# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  InputState *****************************************************
#
"""

Overview
--------

The purpose of an InputState is to receive and combine inputs to a `Mechanism`, allow them to be modified, and provide
them to the Mechanism's `function <Mechanism_Base.function>`. An InputState receives input to a `Mechanism`
provided by the `Projections <Projection>` to that Mechanism from others in a `Process` or `System`.  If the
InputState belongs to an `ORIGIN` Mechanism (see `role of Mechanisms in Processes and Systems
<Mechanism_Role_In_Processes_And_Systems>`), then it receives the input specified when that Process or System is
`run <Run>`.  The `PathwayProjections <PathWayProjection>` received by an InputState are listed in its `path_afferents
<InputState.path_afferents>`, and its `ModulatoryProjections <ModulatoryProjection>` in its `mod_afferents
<InputState.mod_afferents>` attribute.  Its `function <InputState.function>` combines the values received from its
PathWayProjections, modifies the combined value according to value(s) any ModulatoryProjections it receives, and
provides the result to the assigned item of its owner Mechanism's `variable <Mechanism_Base.variable>` and
`input_values <Mechanism_Base.input_values>` attributes (see `below` and `Mechanism InputStates <Mechanism_InputStates>`
for additional details about the role of InputStates in Mechanisms, and their assignment to the items of a Mechanism's
`variable <Mechanism_Base.variable>` attribute).

.. _InputState_Creation:

Creating an InputState
----------------------

An InputState can be created by calling its constructor, but in general this is not necessary as a Mechanism can
usually automatically create the InputState(s) it needs when it is created.  For example, if the Mechanism is
being created within the `pathway <Process.pathway` of a `Process`, its InputState will be created and  assigned
as the `receiver <MappingProjection.receiver>` of a `MappingProjection` from the  preceding `Mechanism <Mechanism>` in
the `pathway <Process.pathway>`.  If it is created using its constructor, and a Mechanism is specified in the
**owner** argument, it is automatically assigned to that Mechanism.  If the **owner* is not specified, `initialization
is deferred.

.. _InputState_Deferred_Initialization:

Owner Assignment and Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An InputState must be owned by a `Mechanism <Mechanism>`.  When InputState is specified in the constructor for a
Mechanism (see `below <InputState_Specification>`), it is automatically assigned to that Mechanism as its owner. If
the InputState is created directly, its `owner <InputState.owner>` can specified in the **owner**  argument of its
constructor, in which case it will be assigned to that Mechanism. Otherwise, its initialization will be
`deferred <State_Deferred_Initialization>` until
COMMENT:
TBI: its `owner <State_Base.owner>` attribute is assigned or
COMMENT
the InputState is assigned to a Mechanism using the Mechanism's `add_states <Mechanism_Base.add_states>` method.

.. _InputState_Primary:

Primary InputState
~~~~~~~~~~~~~~~~~~~

Every Mechanism has at least one InputState, referred to as its *primary InputState*.  If InputStates are not
`explicitly specified <InputState_Specification>` for a Mechanism, a primary InputState is automatically created
and assigned to its `input_state <Mechanism_Base.input_state>` attribute (note the singular), and also to the first
entry of the Mechanism's `input_states <Mechanism_Base.input_states>` attribute (note the plural).  The `value
<InputState.value>` of the primary InputState is assigned as the first (and often only) item of the Mechanism's
`variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>` attributes.

.. _InputState_Specification:

InputState Specification
~~~~~~~~~~~~~~~~~~~~~~~~

Specifying InputStates when a Mechanism is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InputStates can be specified for a `Mechanism <Mechanism>` when it is created, in the **input_states** argument of the
Mechanism's constructor (see `examples <State_Constructor_Argument_Examples>` in State), or in an *INPUT_STATES* entry
of a parameter dictionary assigned to the constructor's **params** argument.  The latter takes precedence over the
former (that is, if an *INPUT_STATES* entry is included in the parameter dictionary, any specified in the
**input_states** argument are ignored).

    .. _InputState_Replace_Default_Note:

    .. note::
       Assigning InputStates to a Mechanism in its constructor **replaces** any that are automatically generated for
       that Mechanism (i.e., those that it creates for itself by default).  If any of those are needed, they must be
       explicitly specified in the list assigned to the **input_states** argument, or the *INPUT_STATES* entry of the
       parameter dictionary in the **params** argument.  The number of InputStates specified must also be equal to
       the number of items in the Mechanism's `variable <Mechanism_Base.variable>` attribute.

.. _InputState_Variable_and_Value:

Each InputState specified in the **input_states** argument must correspond to an item of the Mechanism's `variable
<Mechanism_Base.variable>` attribute (see `Mechanism InputStates <Mechanism_InputStates>`), and the `value
<InputState.value>` of the InputState must be compatible with that item (that is, have the same number and type of
elements).  By default, this is also true of the InputState's `variable <InputState.variable>` attribute,
since the default `function <InputState.function>` for an InputState is a `LinearCombination`, the purpose of which
is to combine the inputs it receives and possibly modify the combined value (under the influence of any
`ModulatoryProjections <ModulatoryProjection>` it receives), but **not mutate its form**. Therefore, under most
circumstances, both the `variable <InputState.variable>` of an InputState and its `value <InputState.value>` should
match the item of its owner's `variable <Mechanism_Base.variable>` to which the InputState is assigned.

The format of an InputState's `variable <InputState.variable>` can be specified in a variety of ways.  The most
straightforward is in the **variable** argument of its constructor.  More commonly, however, it is determined by
the context in which it is being created, such as the specification for its owner Mechanism's `variable
<Mechanism_Base.variable>` or for the InputState in the Mechanism's **input_states** argument (see `below
<InputState_Forms_of_Specification>` and `Mechanism InputState specification <Mechanism_InputState_Specification>`
for details).


Adding InputStates to a Mechanism after it is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InputStates can also be **added** to a Mechanism, either by creating the InputState on its own, and specifying the
Mechanism in the InputState's **owner** argument, or by using the Mechanism's `add_states <Mechanism_Base.add_states>`
method (see `examples <State_Create_State_Examples>` in State).

    .. _InputState_Add_State_Note:

    .. note::
       Adding InputStates *does not replace* any that the Mechanism generates by default;  rather they are added to the
       Mechanism, and appended to the list of InputStates in its `input_states <Mechanism_Base>` attribute. Importantly,
       the Mechanism's `variable <Mechanism_Base.variable>` attribute is extended with items that correspond to the
       `value <InputState.value>` attribute of each added InputState.  This may affect the relationship of the
       Mechanism's `variable <Mechanism_Base.variable>` to its `function <Mechanism_Base.function>`, as well as the
       number of its `OutputStates <OutputState>` (see `note <Mechanism_Add_InputStates_Note>`).

If the name of an InputState added to a Mechanism is the same as one that already exists, its name will be suffixed
with a numerical index (incremented for each InputState with that name; see `Naming`), and the InputState will be
added to the list (that is, it will *not* replace ones that already exist).

.. _InputState_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

InputStates can be specified in a variety of ways, that fall into three broad categories:  specifying an InputState
directly; by an `OutputState` or `Projection` that should project to it; or by using a `State specification dictionary
<State_Specification>` or tuple format to specify attributes for the InputState (including Projections to it). Each of
these is described below:

    .. _InputState_Direct_Specification:

    **Direct Specification**

    * existing **InputState object** or the name of one -- if used to specify an InputState in the constructor for a
      Mechanism, its `value <InputState.value>` must be compatible with the corresponding item of the owner Mechanism's
      `variable <Mechanism_Base.variable>` (see `Mechanism InputState specification
      <Mechanism_InputState_Specification>`).
    ..
    * **InputState class**, **keyword** *INPUT_STATE*, or a **string** -- this creates a default InputState; if used
      to specify an InputState in the constructor for a Mechanism, the item of the owner Mechanism's `variable
      <Mechanism_Base.variable>` to which the InputState is assigned is used as the format for the InputState`s
      `variable <InputState.variable>`; otherwise, the default for the InputState will be used.  If a string is
      specified, it is used as the `name <InputState.name>` of the InputState (see `example
      <State_Constructor_Argument_Examples>`).
    ..
    * **value** -- this creates a default InputState using the specified value as the InputState's `variable
      <InputState.variable>`; if used to specify an InputState in the constructor for a Mechanism, the format must be
      compatible with the corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>` (see
      `Mechanism InputState specification <Mechanism_InputState_Specification>` and `example
      <State_Value_Spec_Example>`).

    .. _InputState_OutputState_Specification:

    **OutputState or Projection Specification**

    * **OutputState** -- a reference to an existing `OutputState`;  this creates an InputState and a
      MappingProjection to it from the specified OutputState.  By default, the OutputState's `value <OutputState.value>`
      is used to format the InputState's `variable <InputState.variable>`, and an `IDENTITY_MATRIX` is used for the
      MappingProjection.  However, if **default_variable** argument is specified in the constructor for the InputState's
      owner Mechanism, then the corresponding item of its value is used as the format for the InputState's `variable
      <InputState.variable>`, and `AUTO_ASSIGN_MATRIX` is used for the MappingProjection, which adjusts its `value
      <Projection_Base.value>` to match the InputState's `variable <InputState.variable>`.

    * **Mechanism** -- the Mechanism's `primary OutputState <OutputState_Primary>` is used, and the InputState
      (and MappingProjection to it) are created as for an OutputState specification (see above).
    ..
    * **Projection object**.  this creates a default InputState and assigns it as the `Projection's <Projection>`
      `receiver <Projection_Base.receiver>`;  the Projection's `value <Projection_Base.value>` must be compatible with
      the item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the InputState is assigned.
    ..
    * **Projection subclass** -- this creates a default InputState, formattig its `variable <InputState.variable>`
      in the same manner as an InputState class, keyword or string (see above), and using this as the format for the
      Projection's `value <Projection_Base.value>`.  Since the Projection's `sender <Projection_Base.sender>` is
      unspecified, its `initialization is deferred <Projection_Deferred_Initialization>`.  In some cases, initialization
      can happen automatically -- for example, when a `ControlProjection` is created for the parameter of a Mechanism
      that is included in a `System`, a `ControlSignal` is created as the Projection's `sender <Projection_Base.sender>`
      that is added to the System's `controller <System.controller>` (see `System Control <System_Control>`).  However,
      for cases in which `deferred initialization <Component_Deferred_Init>` is not automatically completed,
      the Projection will not be operational until its `sender <Projection_Base.sender>` has been specified and its
      initialization completed.

    .. _InputState_Specification_Dictionary:

    **InputState Specification Dictionary or Tuple**

    * **InputState specification dictionary** -- this can be used to specify the attributes of an InputState,
      using any of the entries that can be included in a `State specification dictionary <State_Specification>`
      (see `examples <State_Specification_Dictionary_Examples>` in State), as well as well as the following entries
      specific to an InputState:

      * *WEIGHT*:<number>
          the value must be an integer or float, and is assigned as the value of the InputState's `weight
          <InputState.weight>` attribute (see `weight and exponent <InputState_Weights_And_Exponents>`);
          this takes precedence over any specification in the **weight** argument of the InputState's constructor.
      ..
      * *EXPONENT*:<number>
          the value must be an integer or float, and is assigned as the value of the InputState's `exponent
          <InputState.exponent>` attribute (see `weight and exponent <InputState_Weights_And_Exponents>`);
          this takes precedence over any specification in the **exponent** argument of the InputState's constructor.

      .. _InputState_Projections_Specification:

      If the dictionary is used to specify an InputState in the constructor for a Mechanism, and it includes a
      *VARIABLE* and/or *VALUE* or entry, the value must be compatible with the item of the owner Mechanism's `variable
      <Mechanism_Base.variable>` to which the InputState is assigned (see  `Mechanism InputState specification
      <Mechanism_InputState_Specification>`). If a *PROJECTIONS* entry is included, it can include one or more
      `OutputState` and/or `MappingProjection` specifications for OutputStates that should project to it, as well one
      or more `ModulatorySignal` and/or `ModulatoryProjection` specifications for ModulatorySignals that should
      project to it (see `examples <State_Projections_Examples>` in State).  If these are used to specify the
      InputState's `variable <InputState.variable>` (see `note below <InputState_Projection_Specification>`) within
      the constructor for a Mechanism, then again this must be compatible with the corresponding item of the
      Mechanism's `variable <Mechanism_Base.variable>`.

   .. _InputState_Tuple_Specification:

    * **Tuple specification** -- this is a convenience format that can be used to compactly specify an InputState
      along with a Projection to it.  It can take several forms:

        * **2-item tuple** -- the first item can be either a value (specifying the `variable <InputState.variable>` for
          the InputState, or a Mechanism or OutputState specification indicating an OutputState that should project to
          it (see above); the second item must be a `Projection specification <Projection_Specification>`. This creates
          an InputState and the specified Projection.  If the Projection is a `MappingProjection`, its `value
          <Projection_Base.value>` is used to format of the InputState's `variable <InputState.variable>` (see `note
          <InputState_Projection_Specification>` below); if this occurs within the constructor for a Mechanism, it must
          be compatible with the corresponding item of the Mechanism's `variable <Mechanism_Base.variable>`.  The
          InputState is assigned as the Projection's `receiver <Projection_Base.receiver>` and, if the first item
          specifies an  OutputState, that is assigned as the Projection's `sender <Projection_Base.sender>`.  If the
          specification is for `ModulatoryProjection`, it is created (along with a corresponding `ModulatorySignal`)
          if necessary, the InputState is assigned as its `receiver <ModulatoryProjection.receiver>`, and the the
          Projection is assigned to the InputState's `mod_afferents <InputState>` attribute.

        * **ConnectionTuple** -- this is an expanded version of the 2-item tuple that allows the specification of the
          `weight <InputState.weight>` and/or `exponent <InputState.exponent>` attributes of the InputState, as well as
          a Projection to it. Each tuple must have at least the three following items (in the order listed), and can
          include the fourth:

            * **State specification** -- specifies either the `variable <InputState.variable>` of the InputState, or a
              specification for an OutputState (see above) that should project to it, which must be consistent with
              the Projection specified in the fourth item if that is included (see below);
            |
            * **weight** -- must be an integer or a float; multiplies the `value <InputState.value>` of the InputState
              before it is combined with others by the Mechanism's `function <Mechanism.function>` (see
              ObjectiveMechanism for `examples <ObjectiveMechanism_Weights_and_Exponents_Example>`);
            |
            * **exponent** -- must be an integer or float; exponentiates the `value <InputState.value>` of the
              InputState before it is combined with others by the ObjectiveMechanism's `function
              <ObjectiveMechanism.function>` (see ObjectiveMechanism for `examples
              <ObjectiveMechanism_Weights_and_Exponents_Example>`);
            |
            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` in the
              same manner as the second item of a 2-item tuple (see above);  it's `sender <Projection_Base.sender>`
              must be compatible with the specification in the first item (i.e., for either the `variable
              <InputState.variable>` of the InputState, or the `value <OutputState.value>` of the OutputState
              specified to project to it.

              .. _InputState_Projection_Specification:

              .. note::
                 If a Projection specification is for a `MappingProjection`, it can use the Projection itself or a
                 `matrix specification <Mapping_Matrix_Specification>`.  If it is a Projection, then its `value
                 <Projection_Base.value>` is used to determine the InputState's `variable <InputState.variable>`; if
                 it is a matrix, then its receiver dimensionality determines the format of the InputState's `variable
                 <InputState.variable>`. For a standard 2d weight matrix (i.e., one that maps a 1d array from its
                 `sender <Projection_Base.sender>` to a 1d array of its `receiver <Projection_Base.receiver>`), the
                 receiver dimensionality is its outer dimension (axis 1, or its number of columns).  However, if
                 the `sender <Projection_Base.sender>` has more than one dimension, then the dimensionality of the
                 receiver (used for the InputState's `variable <InputState.variable>`) is the dimensionality of the
                 matrix minus the dimensionality of the sender's `value <OutputState.value>`.

.. _InputState_Projections:

Projections
~~~~~~~~~~~

When an InputState is created, it can be assigned one or more `Projections <Projection>`, using either the
**projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument with
the key *PROJECTIONS*.  An InputState can be assigned either `MappingProjection(s) <MappingProjection>` or
`GatingProjection(s) <GatingProjection>`.  MappingProjections are assigned to its `path_afferents
<InputState.path_afferents>` attribute and GatingProjections to its `mod_afferents <InputState.mod_afferents>`
attribute.  See `State Projections <State_Projections>` for additional details concerning the specification of
Projections when creating a State.


.. _InputState_Structure:

Structure
---------

Every InputState is owned by a `Mechanism <Mechanism>`. It can receive one or more `MappingProjections
<MappingProjection>` from other Mechanisms, as well as from the Process or System to which its owner belongs (if it
is the `ORIGIN` Mechanism for that Process or System).  It has the following attributes, that includes ones specific
to, and that can be used to customize the InputState:

* `projections <OutputState.projections>` -- all of the `Projections <Projection>` received by the InputState.

.. _InputState_Afferent_Projections:

* `path_afferents <InputState.path_afferents>` -- `MappingProjections <MappingProjection>` that project to the
  InputState, the `value <MappingProjection.value>`\\s of which are combined by the InputState's `function
  <InputState.function>`, possibly modified by its `mod_afferents <InputState_mod_afferents>`, and assigned to the
  corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>`.
..
* `mod_afferents <InputState_mod_afferents>` -- `GatingProjections <GatingProjection>` that project to the InputState,
  the `value <GatingProjection.value>` of which can modify the InputState's `value <InputState.value>` (see the
  descriptions of Modulation under `ModulatorySignals <ModulatorySignal_Modulation>` and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the InputState receives more than one GatingProjection,
  their values are combined before they are used to modify the `value <InputState.value>` of InputState.

.. _InputState_Variable:

* `variable <InputState.variable>` -- serves as the template for the `value <Projection_Base.value>` of the
  `Projections <Projection>` received by the InputState:  each must be compatible with (that is, match both the
  number and type of elements of) the InputState's `variable <InputState.variable>`. In general, this must also be
  compatible with the item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the InputState is
  assigned (see `above <InputState_Variable_and_Value>` and `Mechanism InputState
  specification <Mechanism_InputState_Specification>`).

.. _InputState_Function:

* `function <InputState.function>` -- aggregates the `value <Projection_Base.value>` of all of the
  `Projections <Projection>` received by the InputState, and assigns the result to the InputState's `value
  <InputState.value>` attribute.  The default function is `LinearCombination` that performs an elementwise (Hadamard)
  sums the values. However, the parameters of the `function <InputState.function>` -- and thus the `value
  <InputState.value>` of the InputState -- can be modified by any `GatingProjections <GatingProjection>` received by
  the InputState (listed in its `mod_afferents <InputState.mod_afferents>` attribute.  A custom function can also be
  specified, so long as it generates a result that is compatible with the item of the Mechanism's `variable
  <Mechanism_Base.variable>` to which the `InputState is assigned <Mechanism_InputStates>`.

.. _InputState_Value:

* `value <InputState.value>` -- the result returned by its `function <InputState.function>`,
  after aggregating the value of the `PathProjections <PathwayProjection>` it receives, possibly modified by any
  `GatingProjections <GatingProjection>` received by the InputState. It must be compatible with the
  item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the `InputState has been assigned
  <Mechanism_InputStates>` (see `above <InputState_Variable_and_Value>` and `Mechanism InputState specification
  <Mechanism_InputState_Specification>`).

.. _InputState_Weights_And_Exponents:

* `weight <InputState.weight>` and `exponent <InputState.exponent>` -- these can be used by the Mechanism to which the
  InputState belongs when that combines the `value <InputState.value>`\\s of its States (e.g., an ObjectiveMechanism
  uses the weights and exponents assigned to its InputStates to determine how the values it monitors are combined by
  its `function <ObjectiveMechanism>`).  The value of each must be an integer or float, and the default is 1 for both.

.. _InputState_Execution:

Execution
---------

An InputState cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When this occurs, the InputState executes any `Projections <Projection>` it receives, calls its `function
<InputState.function>` to aggregate the values received from any `MappingProjections <MappingProjection>` it receives
(listed in its its `path_afferents  <InputState.path_afferents>` attribute) and modulate them in response to any
`GatingProjections <GatingProjection>` (listed in its `mod_afferents <InputState.mod_afferents>` attribute),
and then assigns the result to the InputState's `value <InputState.value>` attribute. This, in turn, is assigned to
the item of the Mechanism's `variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>`
attributes  corresponding to that InputState (see `Mechanism Variable and InputStates
<Mechanism_Variable_and_InputStates>` for additional details).

.. _InputState_Class_Reference:

Class Reference
---------------

"""
import numbers
import warnings

import numpy as np
import typecheck as tc

from psyneulink.components.component import InitStatus
from psyneulink.components.functions.function import Linear, LinearCombination
from psyneulink.components.mechanisms.mechanism import Mechanism
from psyneulink.components.states.state import \
    StateError, State_Base, _instantiate_state_list, state_type_keywords, ADD_STATES
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.keywords import \
    NAME, DEFERRED_INITIALIZATION, EXPONENT, FUNCTION, INPUT_STATE, INPUT_STATE_PARAMS, MAPPING_PROJECTION, \
    MECHANISM, OUTPUT_STATES, MATRIX, PROJECTIONS, PROJECTION_TYPE, SUM, VARIABLE, WEIGHT, REFERENCE_VALUE, \
    OUTPUT_STATE, PROCESS_INPUT_STATE, SYSTEM_INPUT_STATE, LEARNING_SIGNAL, GATING_SIGNAL, SENDER, COMMAND_LINE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import append_type_to_name, iscompatible, is_numeric

__all__ = [
    'InputState', 'InputStateError', 'state_type_keywords',
]

state_type_keywords = state_type_keywords.update({INPUT_STATE})

# InputStatePreferenceSet = ComponentPreferenceSet(log_pref=logPrefTypeDefault,
#                                                          reportOutput_pref=reportOutputPrefTypeDefault,
#                                                          verbose_pref=verbosePrefTypeDefault,
#                                                          param_validation_pref=paramValidationTypeDefault,
#                                                          level=PreferenceLevel.TYPE,
#                                                          name='InputStateClassPreferenceSet')

# class InputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE

# STATE_SPEC_INDEX = 0 <- DECLARED IN State
WEIGHT_INDEX = 1
EXPONENT_INDEX = 2

class InputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class InputState(State_Base):
    """
    InputState(                                \
    owner,                                     \
    reference_value=None,                      \
    function=LinearCombination(operation=SUM), \
    value=None,                                \
    projections=None,                          \
    weight=None,                               \
    exponent=None,                             \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Subclass of `State <State>` that calculates and represents the input to a `Mechanism <Mechanism>` from one or more
    `PathwayProjection <PathwayProjection>`.

    COMMENT:

        Description
        -----------
            The InputState class is a Component type in the State category of Function,
            Its FUNCTION executes the Projections that it receives and updates the InputState's value

        Class attributes
        ----------------
            + componentType (str) = INPUT_STATE
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination, Operation.SUM)
                + FUNCTION_PARAMS (dict)
                # + kwStateProjectionAggregationFunction (LinearCombination, Operation.SUM)
                # + kwStateProjectionAggregationMode (LinearCombination, Operation.SUM)

        Class methods
        -------------
            _instantiate_function: insures that function is ARITHMETIC)
            update_state: gets InputStateParams and passes to super (default: LinearCombination with Operation.SUM)

        StateRegistry
        -------------
            All INPUT_STATE are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the Mechanism to which the InputState belongs;  it must be specified or determinable from the context in which
        the InputState is created.

    reference_value : number, list or np.ndarray
        the value of the item of the owner Mechanism's `variable <Mechanism_Base.variable>` attribute to which
        the InputState is assigned; used as the template for the InputState's `value <InputState.value>` attribute.

    variable : number, list or np.ndarray
        specifies the template for the InputState's `variable <InputState.variable>` attribute.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to aggregate the `values <Projection_Base.value>` of the `Projections <Projection>`
        received by the InputState, under the possible influence of `GatingProjections <GatingProjection>` received
        by the InputState.  It must produce a result that has the same format (number and type of elements) as the
        item of its owner Mechanism's `variable <Mechanism_Base.variable>` to which the InputState has been assigned.

    projections : list of Projection specifications
        species the `MappingProjection(s) <MappingProjection>` and/or `GatingProjection(s) <GatingProjection>` to be
        received by the InputState, and that will be listed in its `path_afferents <InputState.path_afferents>` and
        `mod_afferents <InputState.mod_afferents>` attributes, respectively (see `InputState_Projections` for additional
        details).

    weight : number : default 1
        specifies the value of the `weight <InputState.weight>` attribute of the InputState.

    exponent : number : default 1
        specifies the value of the `exponent <InputState.exponent>` attribute of the InputState.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the InputState or its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <InputState.name>`
        specifies the name of the InputState; see InputState `name <InputState.name>` for details.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the InputState; see `prefs <InputState.prefs>` for details.


    Attributes
    ----------

    owner : Mechanism
        the Mechanism to which the InputState belongs.

    path_afferents : List[MappingProjection]
        `MappingProjections <MappingProjection>` that project to the InputState
        (i.e., for which it is a `receiver <Projection_Base.receiver>`).

    mod_afferents : List[GatingProjection]
        `GatingProjections <GatingProjection>` that project to the InputState.

    projections : List[Projection]
        all of the `Projections <Projection>` received by the InputState.

    variable : value, list or np.ndarray
        the template for the `value <Projection_Base.value>` of each Projection that the InputState receives,
        each of which must match the format (number and types of elements) of the InputState's
        `variable <InputState.variable>`.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation of the `value <Projection_Base.value>` of each Projection
        received by the InputState, under the possible influence of any `GatingProjections <GatingProjection>` received
        by the InputState.

    value : value or ndarray
        the output of the InputState's `function <InputState.function>`, which is the the aggregated value of the
        `PathwayProjections <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) received by the
        InputState (and listed in its `path_afferents <InputState.path_afferents>` attribute), possibly `modulated
        <ModulatorySignal_Modulation>` by any `GatingProjections <GatingProjection>` it receives (listed in its
        `mod_afferents <InputState.mod_afferents>` attribute).  The result (whether a value or an ndarray) is
        assigned to an item of the owner Mechanism's `variable <Mechanism_Base.variable>`.

    weight : number
        see `weight and exponent <InputState_Weights_And_Exponents>` for description.

    exponent : number
        see `weight and exponent <InputState_Weights_And_Exponents>` for description.

    name : str
        the name of the InputState; if it is not specified in the **name** argument of the constructor, a default is
        assigned by the InputStateRegistry of the Mechanism to which the InputState belongs.  Note that some Mechanisms
        automatically create one or more non-default InputStates, that have pre-specified names.  However, if any
        InputStates are specified in the **input_states** argument of the Mechanism's constructor, those replace those
        InputStates (see `note <Mechanism_Default_State_Suppression_Note>`), and `standard naming conventions <Naming>`
        apply to the InputStates specified, as well as any that are added to the Mechanism once it is created.

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the InputState; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    #region CLASS ATTRIBUTES

    componentType = INPUT_STATE
    paramsType = INPUT_STATE_PARAMS

    stateAttributes = State_Base.stateAttributes | {WEIGHT, EXPONENT}

    connectsWith = [OUTPUT_STATE,
                    PROCESS_INPUT_STATE,
                    SYSTEM_INPUT_STATE,
                    LEARNING_SIGNAL,
                    GATING_SIGNAL]
    connectsWithAttribute = [OUTPUT_STATES]
    projectionSocket = SENDER
    modulators = [GATING_SIGNAL]

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'InputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION,
                               MECHANISM: None,     # These are used to specifiy InputStates by projections to them
                               OUTPUT_STATES: None  # from the OutputStates of a particular Mechanism (see docs)
                               })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=LinearCombination(operation=SUM),
                 projections=None,
                 weight=None,
                 exponent=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        if context is None:
            context = COMMAND_LINE
        else:
            context = self

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  weight=weight,
                                                  exponent=exponent,
                                                  params=params)

        # If owner or reference_value has not been assigned, defer init to State._instantiate_projection()
        # if owner is None or (variable is None and reference_value is None and projections is None):
        if owner is None:
            # Temporarily name InputState
            self._assign_deferred_init_name(name, context)
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = context
            self.init_args['name'] = name
            self.init_args['projections'] = projections

            # Flag for deferred initialization
            self.init_status = InitStatus.DEFERRED_INITIALIZATION
            return

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        # Note: pass name of owner (to override assignment of componentName in super.__init__)
        super(InputState, self).__init__(owner,
                                         variable=variable,
                                         size=size,
                                         projections=projections,
                                         params=params,
                                         name=name,
                                         prefs=prefs,
                                         context=context)

        if self.name is self.componentName or self.componentName + '-' in self.name:
            self._assign_default_state_name(context=context)


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weights and exponents

        This needs to be done here, since paramClassDefault declarations assign None as default
            (so that they can be ignored if not specified here or in the function)
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if WEIGHT in target_set and target_set[WEIGHT] is not None:
            if not isinstance(target_set[WEIGHT], (int, float)):
                raise InputStateError("{} parameter of {} for {} ({}) must be an int or float".
                                      format(WEIGHT, self.name, self.owner.name, target_set[WEIGHT]))

        if EXPONENT in target_set and target_set[EXPONENT] is not None:
            if not isinstance(target_set[EXPONENT], (int, float)):
                raise InputStateError("{} parameter of {} for {} ({}) must be an int or float".
                                      format(EXPONENT, self.name, self.owner.name, target_set[EXPONENT]))

    def _validate_against_reference_value(self, reference_value):
        """Validate that State.value is compatible with reference_value

        reference_value is the item of the owner Mechanism's variable to which the InputState is assigned
        """
        if reference_value is not None and not iscompatible(reference_value, self.value):
            name = self.name or ""
            raise InputStateError("Value specified for {} {} of {} ({}) is not compatible "
                                  "with its expected format ({})".
                                  format(name, self.componentName, self.owner.name, self.value, reference_value))

    def _instantiate_function(self, context=None):
        """Insure that function is LinearCombination and that output is compatible with owner.instance_defaults.variable

        Insures that function:
            - is LinearCombination (to aggregate Projection inputs)
            - generates an output (assigned to self.value) that is compatible with the component of
                owner.function's variable that corresponds to this InputState,
                since the latter will be called with the value of this InputState;

        Notes:
        * Relevant item of owner.function's variable should have been provided
            as reference_value arg in the call to InputState__init__()
        * Insures that self.value has been assigned (by call to super()._validate_function)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """

        super()._instantiate_function(context=context)

        # Insure that function is Function.LinearCombination
        if not isinstance(self.function.__self__, (LinearCombination, Linear)):
            raise StateError("{0} of {1} for {2} is {3}; it must be of LinearCombination or Linear type".
                                      format(FUNCTION,
                                             self.name,
                                             self.owner.name,
                                             self.function.__self__.componentName, ))

        # Insure that self.value is compatible with self.reference_value
        if self.reference_value is not None and not iscompatible(self.value, self.reference_value):
            raise InputStateError("Value ({}) of {} {} for {} is not compatible with specified {} ({})".
                                           format(self.value,
                                                  self.componentName,
                                                  self.name,
                                                  self.owner.name,
                                                  REFERENCE_VALUE,
                                                  self.reference_value))
                                                  # self.owner.variable))

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of State's constructor

        Call _instantiate_projections_to_state to assign:
            PathwayProjections to .path_afferents
            ModulatoryProjections to .mod_afferents
        """
        self._instantiate_projections_to_state(projections=projections, context=context)

    def _execute(self, function_params, context):
        """Call self.function with self._path_proj_values

        If there were no Transmissive Projections, ignore and return None
        """

        # If there were any Transmissive Projections:
        if self._path_proj_values:
            # Combine Projection values
            # TODO: stateful - this seems dangerous with statefulness, maybe safe when self.value is only passed or stateful
            combined_values = self.function(variable=self._path_proj_values,
                                            params=function_params,
                                            context=context)
            return combined_values

        # There were no Projections
        else:
            # mark combined_values as none, so that (after being assigned to self.value)
            #    it is ignored in execute method (i.e., not combined with base_value)
            return None

    def _get_primary_state(self, mechanism):
        return mechanism.input_state

    # def _assign_default_state_name(self, context=None):
    #     # """Assign 'InputState-n' to any InputStates with default name (i.e., name of State: 'InputState'),
    #     #    where n is the next index of InputStates with the default name
    #     # Returns name assigned to State
    #     # """
    #
    #     # Call for State being instantiated in the context of constructing its owner
    #     if isinstance(context,State_Base):
    #         self.name = self.name.replace(self.componentName, 'INPUT_STATE')
    #
    #     # Call in the context of adding a state to an existing owner
    #     elif any(key in context for key in {ADD_STATES, DEFERRED_INITIALIZATION}):
    #         try:
    #             i=len([input_state for input_state in self.owner.input_states if 'InputState-' in input_state.name])
    #             self.name = 'InputState-'+str(i)
    #         except TypeError:
    #             i=0
    #             self.name = 'InputState-'+str(i)
    #
    #     else:
    #         raise InputStateError("PROGRAM ERROR: unrecognized context (\'{}\') for assigning default {} to {}".
    #                               format(context, NAME, InputState.__name__))
    #     return self.name

    @tc.typecheck
    def _parse_state_specific_params(self, owner, state_dict, state_specific_params):
        """Get weights, exponents and/or any connections specified in an InputState specification tuple

        Tuple specification can be:
            (state_spec, connections)
            (state_spec, weights, exponents, connections)

        Returns params dict with WEIGHT, EXPONENT and/or CONNECTIONS entries if any of these was specified.

        """
        # FIX: MAKE SURE IT IS OK TO USE DICT PASSED IN (as params) AND NOT INADVERTENTLY OVERWRITING STUFF HERE

        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputState SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE InputState's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputState VALUES)
        #      THIS WOULD ALLOW AN ADDITONAL HIERARCHICAL LEVEL FOR NESTING ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.components.projections.projection import Projection, _parse_connection_specs

        params_dict = {}
        state_spec = state_specific_params

        if isinstance(state_specific_params, dict):
            # FIX: 10/3/17 - CHECK HERE THAT, IF MECHANISM ENTRY IS USED, A VARIABLE, WEIGHT AND/OR EXPONENT ENTRY
            # FIX:                       IS APPLIED TO ALL THE OutputStates SPECIFIED IN OUTPUT_STATES
            # FIX:                       UNLESS THEY THEMSELVES USE A State specification dict WITH ANY OF THOSE ENTRIES
            # FIX:           USE ObjectiveMechanism EXAMPLES
            return None, state_specific_params

        elif isinstance(state_specific_params, tuple):

            tuple_spec = state_specific_params

            if len(tuple_spec) == 2:
                # FIX: 11/12/17 - ??GENERALIZE FOR ALL STATES AND MOVE TO _parse_state_spec
                # 1st item is a value, so treat as State spec (and return to _parse_state_spec to be parsed)
                if is_numeric(tuple_spec[0]):
                    state_spec = tuple_spec[0]
                    reference_value = state_dict[REFERENCE_VALUE]
                    # Assign value so sender_dim is skipped below
                    # (actual assignment is made in _parse_state_spec)
                    if reference_value is None:
                        state_dict[REFERENCE_VALUE]=state_spec
                    elif  not iscompatible(state_spec, reference_value):
                        raise StateError("Value in first item of 2-item tuple specification for {} of {} ({}) "
                                         "is not compatible with its {} ({})".
                                         format(InputState.__name__, owner.name, state_spec,
                                                REFERENCE_VALUE, reference_value))
                    projections_spec = tuple_spec[1]
                else:
                    # Tuple is projection specification that is used to specify the State,
                    #    so return None in state_spec to suppress further, recursive parsing of it in _parse_state_spec
                    state_spec = None
                    if tuple_spec[0] != self:
                        # If 1st item is not the current state (self), treat as part of the projection specification
                        projections_spec = tuple_spec
                    else:
                        # Otherwise, just use 2nd item as projection spec
                        state_spec = None
                        projections_spec = tuple_spec[1]
            elif len(tuple_spec) == 4:
                # Tuple is projection specification that is used to specify the State,
                #    so return None in state_spec to suppress further, recursive parsing of it in _parse_state_spec
                state_spec = None
                projections_spec = tuple_spec

            if projections_spec is not None:

                try:
                    params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                                       owner=owner,
                                                                       connections=[projections_spec])
                    for projection_spec in params_dict[PROJECTIONS]:
                        # Parse the value of all of the Projections to be comparable to variable of State
                        if state_dict[REFERENCE_VALUE] is None:
                            # FIX: 10/3/17 - PUTTING THIS HERE IS A HACK...
                            # FIX:           MOVE TO _parse_state_spec UNDER PROCESSING OF ConnectionTuple SPEC
                            # FIX:           USING _get_state_for_socket
                            # from psyneulink.components.projections.projection import _parse_projection_spec
                            sender_dim = projection_spec.state.value.ndim
                            projection = projection_spec.projection
                            if isinstance(projection, dict):
                                matrix = projection[MATRIX]
                            elif isinstance(projection, Projection):
                                if projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                                    continue
                                matrix = projection.matrix
                            else:
                                raise InputStateError("Unrecognized Projection specification for {} of {} ({})".
                                                      format(self.name, owner.name, projection_spec))
                            # Remove dimensionality of sender OutputState, and assume that is what receiver will receive
                            proj_val_shape = matrix.shape[sender_dim :]
                            state_dict[VARIABLE] = np.zeros(proj_val_shape)

                except InputStateError:
                    raise InputStateError("Tuple specification in {} specification dictionary "
                                          "for {} ({}) is not a recognized specification for one or more "
                                          "{}s, {}s, or {}s that project to it".
                                          format(InputState.__name__,
                                                 owner.name,
                                                 projections_spec,
                                                 Mechanism.__name__,
                                                 OutputState.__name__,
                                                 Projection.__name__))

            # Get weights and exponents if specified
            if len(tuple_spec) == 2:
                pass

            # Tuple is (spec, weights, exponents<, afferent_source_spec>),
            #    for specification of weights and exponents,  + connection(s) (afferent projection(s)) to InputState
            elif len(tuple_spec) in {3, 4}:

                weight = tuple_spec[WEIGHT_INDEX]
                exponent = tuple_spec[EXPONENT_INDEX]

                if weight is not None and not isinstance(weight, numbers.Number):
                    raise InputStateError("Specification of the weight ({}) in tuple of {} specification dictionary "
                                          "for {} must be a number".format(weight, InputState.__name__, owner.name))
                params_dict[WEIGHT] = weight

                if exponent is not None and not isinstance(exponent, numbers.Number):
                    raise InputStateError("Specification of the exponent ({}) in tuple of {} specification dictionary "
                                          "for {} must be a number".format(exponent, InputState.__name__, owner.name))
                params_dict[EXPONENT] = exponent

            else:
                raise StateError("Tuple provided as state_spec for {} of {} ({}) must have either 2, 3 or 4 items".
                                 format(InputState.__name__, owner.name, tuple_spec))

        elif state_specific_params is not None:
            raise InputStateError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, state_specific_params))

        return state_spec, params_dict

    @property
    def pathway_projections(self):
        return self.path_afferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.path_afferents = assignment


def _instantiate_input_states(owner, input_states=None, reference_value=None, context=None):
    """Call State._instantiate_state_list() to instantiate ContentAddressableList of InputState(s)

    Create ContentAddressableList of InputState(s) specified in paramsCurrent[INPUT_STATES]

    If input_states is not specified:
        - use owner.input_states as list of InputState specifications
        - if owner.input_states is empty, user owner.instance_defaults.variable to create a default InputState

    When completed:
        - self.input_states contains a ContentAddressableList of one or more input_states
        - self.input_state contains the `primary InputState <InputState_Primary>`:  first or only one in input_states
        - paramsCurrent[INPUT_STATES] contains the same ContentAddressableList (of one or more input_states)
        - each InputState corresponds to an item in the variable of the owner's function
        - the value of all of the input_states is stored in a list in input_value
        - if there is only one InputState, it is assigned the full value

    Note: State._instantiate_state_list()
              parses self.instance_defaults.variable (2D np.array, passed in reference_value)
              into individual 1D arrays, one for each input state

    (See State._instantiate_state_list() for additional details)

    Returns list of instantiated InputStates
    """

    # This allows method to be called by Mechanism.add_input_states() with set of user-specified input_states,
    #    while calls from init_methods continue to use owner.input_states (i.e., InputState specifications
    #    assigned in the **input_states** argument of the Mechanism's constructor)
    input_states = input_states or owner.input_states

    state_list = _instantiate_state_list(owner=owner,
                                         state_list=input_states,
                                         state_type=InputState,
                                         state_param_identifier=INPUT_STATE,
                                         reference_value=reference_value if reference_value is not None
                                                                         else owner.instance_defaults.variable,
                                         reference_value_name=VARIABLE,
                                         context=context)

    # Call from Mechanism.add_states, so add to rather than assign input_states (i.e., don't replace)
    if context and 'ADD_STATES' in context:
        owner.input_states.extend(state_list)
    else:
        owner._input_states = state_list

    # Check that number of input_states and their variables are consistent with owner.instance_defaults.variable,
    #    and adjust the latter if not
    variable_item_is_OK = False
    for i, input_state in enumerate(owner.input_states):
        try:
            variable_item_is_OK = iscompatible(owner.instance_defaults.variable[i], input_state.value)
            if not variable_item_is_OK:
                break
        except IndexError:
            variable_item_is_OK = False
            break

    if not variable_item_is_OK:
        # NOTE: This block of code appears unused, and the 'for' loop appears to cause an error anyways. (7/11/17 CW)
        old_variable = owner.instance_defaults.variable
        new_variable = []
        for state in owner.input_states:
            new_variable.append(state.value)
        owner.instance_defaults.variable = np.array(new_variable)
        if owner.verbosePref:
            warnings.warn(
                "Variable for {} ({}) has been adjusted to match number and format of its input_states: ({})".format(
                    old_variable,
                    append_type_to_name(owner),
                    owner.instance_defaults.variable,
                )
            )

    return state_list
