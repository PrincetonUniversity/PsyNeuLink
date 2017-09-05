# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#  *********************************************  State ********************************************************

"""

Overview
--------

A State provides an interface to one or more `Projections <Projection>`, and receives the `value(s) <Projection>`
provide by them.  The value of a State can be modulated by a `ModulatoryProjection <ModulatoryProjection>`. There are
three primary types of States, all of which are used by `Mechanisms <Mechanism>`, one of which is used by
`MappingProjections <MappingProjection>`, and that are subject to modulation by `ModulatorySignals <ModulatorySignal>`,
as summarized in the table below:

+-------------------+--------------------+------------------------+--------------------+-------------------------------+
| *State Type*      | *Owner*            |      *Description*     |    *Modulated by*  |       *Specification*         |
+===================+====================+========================+====================+===============================+
| `InputState`      |  `Mechanism        |receives input from     | `GatingSignal`     |`InputState` constructor;      |
|                   |  <Mechanism>`      |`MappingProjection`     |                    |`Mechanism <Mechanism>`        |
|                   |                    |                        |                    |constructor or its             |
|                   |                    |                        |                    |`add_states` method            |
+-------------------+--------------------+------------------------+--------------------+-------------------------------+
|`ParameterState`   |  `Mechanism        |represents parameter    |`LearningSignal`    |Implicitly whenever a          |
|                   |  <Mechanism>` or   |value for a `Component  |and/or              |parameter value is             |
|                   |  `Projection       |<Component>`            |`ControlSignal`     |`specified                     |
|                   |  <Projection>`     |or its function         |                    |<ParameterState_Specification>`|
+-------------------+--------------------+------------------------+--------------------+-------------------------------+
| `OutputState`     |  `Mechanism        |provides output to      | `GatingSignal`     |`OutputState` constructor;     |
|                   |  <Mechanism>`      |`MappingProjection`     |                    |`Mechanism <Mechanism>`        |
|                   |                    |                        |                    |constructor or its             |
|                   |                    |                        |                    |`add_states` method            |
+-------------------+--------------------+------------------------+--------------------+-------------------------------+
|`ModulatorySignal  |`AdaptiveMechanism  |provides value for      |                    |`AdaptiveMechanism             |
|<ModulatorySignal>`|<AdaptiveMechanism>`|`ModulatoryProjection   |                    |<AdaptiveMechanism>`           |
|                   |                    |<ModulatoryProjection>` |                    |constructor; tuple in State    |
|                   |                    |                        |                    |or parameter specification     |
+-------------------+--------------------+------------------------+--------------------+-------------------------------+

COMMENT:

* `InputState`:
    used by a Mechanism to receive input from `MappingProjections <MappingProjection>`;
    its value can be modulated by a `GatingSignal`.

* `ParameterState`:
    * used by a Mechanism to represent the value of one of its parameters, or a parameter of its
      `function <Mechanism_Base.function>`, that can be modulated by a `ControlSignal`;
    * used by a `MappingProjection` to represent the value of its `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, that can be modulated by a `LearningSignal`.

* `OutputState`:
    used by a Mechanism to send its value to any efferent projections.  For
    `ProcessingMechanisms <ProcessingMechanism>` these are `PathwayProjections <PathwayProjection>`, most commonly
    `MappingProjections <MappingProjection>`.  For `ModulatoryMechanisms <ModulatoryMechanism>`, these are
    `ModulatoryProjections <ModulatoryProjection>` as described below. The `value <OutputState.value>` of an
    OutputState can be modulated by a `GatingSignal`.

* `ModulatorySignal <ModulatorySignal>`:
    a subclass of `OutputState` used by `AdaptiveMechanisms <AdaptiveMechanism>` to modulate the value of the primary
    types of States listed above.  There are three types of ModulatorySignals:

    * `LearningSignal`, used by a `LearningMechanism` to modulate the *MATRIX* ParameterState of a `MappingProjection`;
    * `ControlSignal`, used by a `ControlMechanism <ControlMechanism>` to modulate the `ParameterState` of a `Mechanism
      <Mechanism>`;
    * `GatingSignal`, used by a `GatingMechanism` to modulate the `InputState` or `OutputState` of a `Mechanism
       <Mechanism>`.
    Modulation is discussed further `below <State_Modulation>`, and described in detail under
    `ModulatorySignals <ModulatorySignal_Modulation>`.

COMMENT

.. _State_Creation:

Creating a State
----------------

In general, States are created automatically by the objects to which they belong (their `owner <State_Owner>`),
or by specifying the State in the constructor for its owner.  For example, unless otherwise specified, when a
`Mechanism <Mechanism>` is created it creates a default `InputState` and `OutputState` for itself, and whenever
any Component is created, it automatically creates a `ParameterState` for each of its `configurable parameters
<Component_Configurable_Attributes>` and those of its `function <Component_Function>`. States are also created in
response to explicit specifications.  For example, InputStates and OutputStates can be specified in the constructor
for a Mechanism (see `Mechanism_State_Specification`) or in its `add_states` method; and a ParameterState is
specified in effect when the value of a parameter for any Component or its `function <Component.function>` is
specified in its constructor.  InputStates and OutputStates (but not ParameterStates) can also be created directly
using their constructors;  however, Parameter States cannot be created in this way; they are always and only created
when the Component to which a parameter belongs is created.

.. _State_Specification:

Specifying a State
~~~~~~~~~~~~~~~~~~

Wherever a State is specified, it can be done using any of the following:

    * an existing **State** object;
    ..
    * the name of a **State subclass** (`InputState`, `ParameterState`, or `OutputState`) - a default State of the
      corresponding type will be created, using a default value for the State that is determined by the context
      in which it is specified.
    ..
    * a **value**.  This creates a default State using the specified value as its default `value <State_Base.value>`.
    ..
    * a **State specification dictionary**; every State specification can contain the following: *KEY*:<value>
      entries, in addition to those specific to a particular State subtype (see subtype documentation):

      ..
      * *NAME*:<str>
          the string is used as the name of the State;

      ..
      * *STATE_TYPE*:<State type>
          specifies type of State to create (necessary if it cannot be determined from the
          the context of the other entries or in which it is being created)
      ..
      * *VALUE*:<value>
          the value is used as the default value of the State;
      ..
      * *PROJECTIONS*:<List>
          the list must contain specifications for one or more
          `Projections <Projection_In_Context_Specification>` to or from the State, and/or
          `ModulatorySignals <ModulatorySignal>` from which it should receive projections;
          the type of Projections it can send and/or receive depends the type of State and
          the context in which it is specified (see `State_Projections` below);
      ..
      * *str*:<List>
          the key is used as the name of the State, and the list must contain specifications for
          one or more `Projections <Projection_In_Context_Specification>` to or from the State
          depending on the type of State and the context in which it is specified;
        ..

    * a **2-item tuple** - the first item must be a value, used as the default value for the State,
      and the second item must be a specification for a `Projection <Projection_In_Context_Specification>`
      to or from the State, depending on the type of State and the context in which it is specified;

.. _State_Deferred_Initialization:

`InputStates <InputState>`, `OutputStates <OutputState>` and `ModulatorySignals <ModulatorySignal>` can also be
created on their own, by using the relevant constructors;  however, `ParameterStates <ParameterState>` cannot be
created on their own. If a State is created on its own, and its `owner <State_Owner>` is not specified, then its
initialization will be `deferred <Component_Deferred_Initialization>`.  Its initialization is completed automatically
when it is assigned to an owner `Mechanism <Mechanism_Base>` using the owner's `add_states` method.  If the State is
not assigned to an owner, it will not be functional (i.e., used during the execution of `Mechanisms
<Mechanism_Base_Execution>` and/or `Compositions <Composition_Execution>`, irrespective of whether it has any
`Projections <Projection>` assigned to it.

.. _State_Projections:

Projections
~~~~~~~~~~~

When a State is created, it can be assigned one or more `Projections <Projection>`, using either the **projections**
argument of its constructor, or in an entry of a dictionary assigned to the **params** argument with the key
*PROJECTIONS*. The following of types of Projections can be specified for each type of State:

    * `InputState`
        • `PathwayProjection(s) <PathwayProjection>`
          - assigned to its `pathway_afferents <Input.pathway_afferents>` attribute.
        • `GatingProjection(s) <GatingProjection>`
          - assigned to its `mod_afferents <InputState.mod_afferents>` attribute.

    * `ParameterState`
        • `ControlProjection(s) <ControlProjection>` - assigned to its `mod_afferents <ParameterState.mod_afferents>`
          attribute.

    * `OutputState`
        • `PathwayProjection(s) <PathwayProjection>`
          - assigned to its `efferents <Output.efferents>` attribute.
        • `GatingProjection(s) <GatingProjection>`
          - assigned to its `mod_afferents <OutputState.mod_afferents>` attribute.

    * `ModulatorySignal <ModulatorySignal>`
        • `ModulatoryProjection(s) <ModulatoryProjection>`
          - assigned to its `efferents <ModulatorySignal.efferents>` attribute.

Projections must be specified in a list.  Each entry must be either a specification for a `projection
<Projection_In_Context_Specification>`, or for a `sender <Projection.sender>` or `receiver <Projection.receiver>`, in
which case the appropriate type of Projection is created.  A sender or receiver can be specified as a `State <State>`
or a `Mechanism <Mechanism>`. If a Mechanism is specified, its primary `InputState <InputState_Primary>` or `OutputState
<OutputState_Primary>`  is used, as appropriate.  When a sender or receiver is used to specify the Projection, the type
of Projection created is inferred from the State and the type of sender or receiver specified, as illustrated in the
examples below.  Note that the State must be `assigned to an owner <State_Creation>` in order to be functional,
irrespective of whether any `Projections <Projection>` have been assigned to it.

COMMENT:
    ADD TABLE HERE SHOWING COMBINATIONS OF ALLOWABLE SPECIFICATIONS AND THEIR OUTCOMES
COMMENT

Examples
^^^^^^^^

The following creates an InputState ``my_input_state`` with a `MappingProjection` to it from the
`primary OutputState <OutputState_Primary>` of ``mech_A``::

    my_input_state = InputState(projections=[mech_A])

The following creates a `GatingSignal` with `GatingProjections <GatingProjection>` to ``mech_B`` and ``mech_C``,
and assigns it to a ``my_gating_mech``::

    my_gating_signal = GatingSignal(projections=[mech_B, mech_C])
    my_gating_mech = GatingMechanism(gating_signals=[my_gating_signal]

The GatingMechanism created will now gate the `primaryInputStates <Mechanism_InputStates>` of ``mech_B`` and ``mech_C``.

.. _State_Structure:

Structure
---------

.. _State_Owner:

Owner
~~~~~

Every State has an `owner <State_Base.owner>`.  For `InputStates <InputState>` and `OutputStates <OutputState>`, the
owner must be a `Mechanism <Mechanism>`.  For `ParameterStates <ParameterState>` it can be a Mechanism or a
`PathwayProjection <PathwayProjection>`. For `ModulatorySignals <ModulatorySignal>`, it must be an `AdaptiveMechanism
<AdaptiveMechanism>`. When a State is created as part of another Component, its `owner <State_Base.owner>` is
assigned automatically to that Component.  It is also assigned automatically when the State is assigned to a
`Mechanism <Mechanism>` using that Mechanism's `add_states` method.  Otherwise, it must be specified explicitly in
the **owner** argument of the constructor for the State.  If it is not, the State's initialization will be `deferred
<State_Deferred_Initialization>` until it has been assigned to an owner.

Variable, Function and Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition, like all PsyNeuLink Components, it also has the three following core attributes:

    * `variable <State_Base.variable>`:  for an `InputState` and `ParameterState`,
      the value of this is determined by the value(s) of the Projection(s) that it receives (and that are listed in
      its `path_afferents <State_Base.path_afferents>` attribute).  For an `OutputState`, it is the item of the owner
      Mechanism's `value <Mechanism_Base.value>` to which the OutputState is assigned (specified by the OutputState's
      `index <OutputState_Index>` attribute.
    ..
    * `function <State_Base.function>`:  for an `InputState` this aggregates the values of the Projections that the
      State receives (the default is `LinearCombination` that sums the values), under the potential influence of a
      `GatingSignal`;  for a `ParameterState`, it determines the value of the associated parameter, under the
      potential influence of a `ControlSignal` (for a `Mechanism <Mechanism>`) or a `LearningSignal` (for a
      `MappingProjection`); for an OutputState, it conveys the result  of the Mechanism's function to its
      `output_values <Mechanism_Base.output_values>` attribute, under the potential influence of a `GatingSignal`.  See
      `ModulatorySignals <ModulatorySignal_Structure>` and the `AdaptiveMechanism <AdaptiveMechanism>` associated with
      each type for a description of how they can be used to modulate the `function <State_Base.function>` of a State.
    ..
    * `value <State_Base.value>`:  for an `InputState` this is the aggregated value of the `PathwayProjections` it
      receives;  for a `ParameterState`, this represents the value of the parameter that will be used by the
      State's owner or its `function <Component.function>`; for an `OutputState`, it is the item of the  owner
      Mechanism's `value <Mechanisms.value>` to which the OutputState is assigned, possibly modified by its
      `calculate <OutputState_Calculate>` attribute and/or a `GatingSignal`, and used as the
      `value <Projection.value>` of the projections listed in its `efferents <OutputState.path_efferents>` attribute.

.. _State_Modulation:

Modulation
~~~~~~~~~~

Every type of State has a `mod_afferents <State_Base.mod_afferents>` attribute, that lists the `ModulatoryProjections
<ModulatoryProjection>` it receives.  Each ModulatoryProjection comes from a `ModulatorySignal <ModulatorySignal>`
that specifies how it should modulate the State's `value <State_Base.value>` when the State is updated (see
`ModulatorySignal_Modulation` and `ModulatorySignal_Anatomy_Figure`).  In most cases, a ModulatorySignal uses the
State's `function <State_Base.function>` to modulate its `value <State_Base.value>`.  The function of every State
assigns one of its parameters as its *ADDITIVE_PARAM* and another as its *MULTIPLICATIVE_PARAM*. The
`modulation <ModulatorySignal.modulation>` attribute of a ModulatorySignal determines which of these to modify when the
State uses it `function <State_Base.function>` to calculate its `value  <State_Base.value>`.  However, the
ModulatorySignal can also be configured to override the State's `value <State_Base.value>` (i.e., assign it directly),
or to disable modulation, using one of the values of `ModulationParam` for its
`modulation <ModulatorySignal.modulation>` attribute (see `ModulatorySignal_Modulation` for a more detailed discussion).

.. _State_Execution:

Execution
---------

States cannot be executed.  They are updated when the Component to which they belong is executed.  InputStates and
ParameterStates belonging to a Mechanism are updated before the Mechanism's function is called.  OutputStates are
updated after the Mechanism's function is called.  When a State is updated, it executes any Projections that project
to it (listed in its `all_afferents <State_Base.all_afferents>` attribute.  It uses the values it receives from any
`PathWayProjections` (listed in its `path_afferents` attribute) as the variable for its
`function <State_Base.function>`. It then executes all of the ModulatoryProjections it receives.  Different
ModulatorySignals may call for different forms of modulation (see `ModulatorySignal_Modulation`).  Accordingly,
it separately sums the values specified by any ModulatorySignals for the *MULTIPLICATIVE_PARAM* of its
`function <State_Base.function>`, and similarly for the *ADDITIVE_PARAM*.  It then applies the summed value for each
to the corresponding parameter of its `function <State_Base.function>`.  If any of the ModulatorySignals specifies
*OVERRIDE*, then the value of that ModulatorySignal is used as the State's `value <State_Base.value>`. Finally,
the State calls its `function <State_Base.function>` to determine its `value <State_Base.value>`.

.. note::
   The change in the value of a `State <State>` does not occur until the Mechanism to which the State belongs is next
   executed; This conforms to a "lazy evaluation" protocol (see :ref:`Lazy Evaluation <LINK>` for an explanation
   of "lazy" updating).

.. _State_Class_Reference:

Class Reference
---------------

"""

import inspect
import numbers
import warnings

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import Component, ComponentError, InitStatus, component_keywords, function_type
from PsyNeuLink.Components.Functions.Function import LinearCombination, ModulationParam, _get_modulated_param, get_param_value_for_function, get_param_value_for_keyword
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.ShellClasses import Mechanism, Process, Projection, State
from PsyNeuLink.Globals.Keywords import VARIABLE, SIZE, FUNCTION_PARAMS, NAME, OWNER, PARAMS, CONTEXT, EXECUTING, \
    VALUE, STATE, STATE_PARAMS, STATE_TYPE, STATE_VALUE, \
    STANDARD_ARGS, STANDARD_OUTPUT_STATES, \
    PROJECTIONS, PROJECTION_PARAMS,  PROJECTION_TYPE, RECEIVER, SENDER, \
    MAPPING_PROJECTION_PARAMS, MATRIX, MATRIX_KEYWORD_SET, \
    MODULATION, MODULATORY_SIGNAL, \
    LEARNING, LEARNING_PROJECTION, LEARNING_PROJECTION_PARAMS, LEARNING_SIGNAL_SPECS, \
    CONTROL, CONTROL_PROJECTION, CONTROL_PROJECTION_PARAMS, CONTROL_SIGNAL_SPECS, \
    GATING, GATING_PROJECTION, GATING_PROJECTION_PARAMS, GATING_SIGNAL_SPECS, INITIALIZING, \
    kwAssign, kwStateComponentCategory, kwStateContext, kwStateName, kwStatePrefs
from PsyNeuLink.Globals.Log import LogEntry, LogLevel
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import kpVerbosePref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.Utilities import ContentAddressableList, MODULATION_OVERRIDE, Modulation, append_type_to_name, convert_to_np_array, get_class_attributes, is_value_spec, iscompatible, merge_param_dicts, type_match
from PsyNeuLink.Scheduling.TimeScale import CurrentTime, TimeScale

state_keywords = component_keywords.copy()
state_keywords.update({STATE_VALUE,
                       STATE_PARAMS,
                       PROJECTIONS,
                       PROJECTION_TYPE,
                       LEARNING_PROJECTION_PARAMS,
                       LEARNING_SIGNAL_SPECS,
                       CONTROL_PROJECTION_PARAMS,
                       CONTROL_SIGNAL_SPECS,
                       GATING_PROJECTION_PARAMS,
                       GATING_SIGNAL_SPECS
                       })

state_type_keywords = {STATE_TYPE}


def _is_state_type (spec):
    if issubclass(spec, State):
        return True
    return False

# Note:  This is created only for assignment of default projection types for each State subclass (see .__init__.py)
#        Individual stateRegistries (used for naming) are created for each Mechanism
StateRegistry = {}

class StateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# State factory method:
# def state(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of State
#
#        If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterState)
#         If called with a name string:
#             - if registered in owner Mechanism's state_registry as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in Mechanism's _stateRegistry:
#             return _stateRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultState
#             return DefaultState(name, params)



# DOCUMENT:  INSTANTATION CREATES AN ATTIRBUTE ON THE OWNER MECHANISM WITH THE STATE'S NAME + kwValueSuffix
#            THAT IS UPDATED BY THE STATE'S value setter METHOD (USED BY LOGGING OF MECHANISM ENTRIES)
class State_Base(State):
    """
    State_Base(        \
    owner,             \
    variable=None,     \
    size=None,         \
    projections=None,  \
    params=None,       \
    name=None,         \
    prefs=None)

    Base class for State.

    .. note::
       State is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <State_Subtypes>`.


    COMMENT:
        Description
        -----------
            Represents and updates the value of the input, output or parameter of a Mechanism
                - receives inputs from projections (self.path_afferents, PROJECTIONS)
                - input_states and parameterStates: combines inputs from all projections (mapping, control or learning)
                    and uses this as variable of function to update the value attribute
                - output_states: represent values of output of function
            Value attribute:
                 - is updated by the execute method (which calls State's function)
                 - can be used as sender (input) to one or more projections
                 - can be accessed by KVO
            Constraints:
                - value must be compatible with variable of function
                - value must be compatible with receiver.value for all projections it receives

            Subclasses:
                Must implement:
                    componentType
                    ParamClassDefaults with:
                        + FUNCTION (or <subclass>.function
                        + FUNCTION_PARAMS (optional)
                        + PROJECTION_TYPE - specifies type of projection to use for instantiation of default subclass
                Standard subclasses and constraints:
                    InputState - used as input State for Mechanism;  additional constraint:
                        - value must be compatible with variable of owner's function method
                    OutputState - used as output State for Mechanism;  additional constraint:
                        - value must be compatible with the output of the owner's function
                    MechanismsParameterState - used as State for Mechanism parameter;  additional constraint:
                        - output of function must be compatible with the parameter's value

        Class attributes
        ----------------
            + componentCategory = kwStateFunctionCategory
            + className = STATE
            + suffix
            + classPreference (PreferenceSet): StatePreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + ClassDefaults.variable (value): [0]
            + requiredParamClassDefaultTypes = {FUNCTION_PARAMS : [dict],    # Subclass function params
                                               PROJECTION_TYPE: [str, Projection]})   # Default projection type
            + paramClassDefaults (dict): {PROJECTIONS: []}             # Projections to States
            + owner (Mechansim)
            + FUNCTION (Function class or object, or method)

        Class methods
        -------------
            - set_value(value) -
                validates and assigns value, and updates observers
                returns None
            - update_state(context) -
                updates self.value by combining all projections and using them to compute new value
                return None

        StateRegistry
        -------------
            Used by .__init__.py to assign default projection types to each State subclass
            Note:
            * All States that belong to a given owner are registered in the owner's _stateRegistry,
                which maintains a dict for each State type that it uses, a count for all instances of that type,
                and a dictionary of those instances;  NONE of these are registered in the StateRegistry
                This is so that the same name can be used for instances of a State type by different owners
                    without adding index suffixes for that name across owners,
                    while still indexing multiple uses of the same base name within an owner

        Arguments
        ---------
        - value (value) - establishes type of value attribute and initializes it (default: [0])
        - owner(Mechanism) - assigns State to Mechanism (default: NotImplemented)
        - params (dict):  (if absent, default State is implemented)
            + FUNCTION (method)         |  Implemented in subclasses; used in update()
            + FUNCTION_PARAMS (dict) |
            + PROJECTIONS:<projection specification or list of ones>
                if absent, no projections will be created
                projection specification can be: (see Projection for details)
                    + Projection object
                    + Projection class
                    + specification dict
                    + a list containing any or all of the above
                    if dict, must contain entries specifying a projection:
                        + PROJECTION_TYPE:<Projection class>: must be a subclass of Projection
                        + PROJECTION_PARAMS:<dict>? - must be dict of params for PROJECTION_TYPE
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): must be a reference to a subclass, or an exception will be raised
    COMMENT

    Attributes
    ----------

    owner : Mechanism or Projection
        object to which the State belongs (see `State_Owner` for additional details).

    base_value : number, list or np.ndarray
        value with which the State was initialized.

    all_afferents : Optional[List[Projection]]
        list of all Projections received by the State (i.e., for which it is a `receiver <Projection.receiver>`.

    path_afferents : Optional[List[Projection]]
        list all `PathwayProjections <PathwayProjection>` received by the State.
        (note:  only `InputStates <InputState>` have path_afferents;  the list is empty for other types of States).

    mod_afferents : Optional[List[GatingProjection]]
        list of all `ModulatoryProjections <ModulatoryProjection>` received by the State.

    efferents : Optional[List[Projection]]
        list of outgoing Projections from the State (i.e., for which is a `sender <Projection.sender>`
        (note:  only `OutputStates <OutputState>` have efferents;  the list is empty for other types of States).

    function : TransferFunction : default determined by type
        used to determine the State's own value from the value of the Projection(s) it receives;  the parameters that
        the TransferFunction identifies as ADDITIVE and MULTIPLICATIVE are subject to modulation by a
        `ModulatoryProjection <ModulatoryProjection_Structure>`.

    value : number, list or np.ndarray
        current value of the State (updated by `update <State_Base.update>` method).

    name : str : default <State subclass>-<index>
        the name of the State.
        Specified in the **name** argument of the constructor for the State;
        if not specified, a default is assigned by StateRegistry based on the
        States's subclass (see :doc:`Registry <LINK>` for conventions used in naming,
        including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink Components, States names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation).

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the State.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentCategory = kwStateComponentCategory
    className = STATE
    suffix = " " + className
    paramsType = None

    class ClassDefaults(State.ClassDefaults):
        variable = [0]

    registry = StateRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    requiredParamClassDefaultTypes = Component.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({FUNCTION_PARAMS : [dict],
                                           PROJECTION_TYPE: [str, Projection]})   # Default projection type
    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({STATE_TYPE: None})

    @tc.typecheck
    def __init__(self,
                 owner:tc.any(Mechanism, Projection),
                 variable=None,
                 size=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kargs):
        """Initialize subclass that computes and represents the value of a particular State of a Mechanism

        This is used by subclasses to implement the InputState(s), OutputState(s), and ParameterState(s) of a Mechanism.

        Arguments:
            - owner (Mechanism):
                 Mechanism with which State is associated (default: NotImplemented)
                 this argument is required, as can't instantiate a State without an owning Mechanism
            - variable (value): value of the State:
                must be list or tuple of numbers, or a number (in which case it will be converted to a single-item list)
                must match input and output of State's update function, and any sending or receiving projections
            - size (int or array/list of ints):
                Sets variable to be array(s) of zeros, if **variable** is not specified as an argument;
                if **variable** is specified, it takes precedence over the specification of **size**.
            - params (dict):
                + if absent, implements default State determined by PROJECTION_TYPE param
                + if dict, can have the following entries:
                    + PROJECTIONS:<Projection object, Projection class, dict, or list of either or both>
                        if absent, no projections will be created
                        if dict, must contain entries specifying a projection:
                            + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                            + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            - name (str): string with name of State (default: name of owner + suffix + instanceIndex)
            - prefs (dict): dictionary containing system preferences (default: Prefs.DEFAULTS)
            - context (str)
            - **kargs (dict): dictionary of arguments using the following keywords for each of the above kargs:
                # + STATE_VALUE = value
                + VARIABLE = variable
                + STATE_PARAMS = params
                + kwStateName = name
                + kwStatePrefs = prefs
                + kwStateContext = context
                NOTES:
                    * these are used for dictionary specification of a State in param declarations
                    * they take precedence over arguments specified directly in the call to __init__()
        """
        if kargs:
            try:
                variable = self._update_variable(kargs[VARIABLE])
            except (KeyError, NameError):
                pass
            try:
                size = kargs[SIZE]
            except (KeyError, NameError):
                pass
            try:
                projections = kargs[PROJECTIONS]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[kwStateName]
            except (KeyError, NameError):
                pass
            try:
                prefs = kargs[kwStatePrefs]
            except (KeyError, NameError):
                pass
            try:
                context = kargs[kwStateContext]
            except (KeyError, NameError):
                pass

        # Enforce that only called from subclass
        if not isinstance(context, State_Base):
            raise StateError("Direct call to abstract class State() is not allowed; "
                                      "use state() or one of the following subclasses: {0}".
                                      format(", ".join("{!s}".format(key) for (key) in StateRegistry.keys())))

        # Enforce that subclass must implement and _execute method
        if not hasattr(self, '_execute'):
            raise StateError("{}, as a subclass of {}, must implement an _execute() method".
                             format(self.__class__.__name__, STATE))

        # MODIFIED 7/12/17 OLD:
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(projections=projections,
                                                  params=params)

        self.owner = owner

        # Register State with StateRegistry of owner (Mechanism to which the State is being assigned)
        register_category(entry=self,
                          base_class=State_Base,
                          name=name,
                          registry=owner._stateRegistry,
                          # sub_group_attr='owner',
                          context=context)

        self.path_afferents = []
        self.mod_afferents = []
        self.efferents = []
        self._stateful = False

        self._path_proj_values = []
        # Create dict with entries for each ModualationParam and initialize - used in update() to coo
        self._mod_proj_values = {}
        for attrib, value in get_class_attributes(ModulationParam):
            self._mod_proj_values[getattr(ModulationParam,attrib)] = []

        # VALIDATE VARIABLE, PARAM_SPECS, AND INSTANTIATE self.function
        super(State_Base, self).__init__(default_variable=variable,
                                         size=size,
                                         param_defaults=params,
                                         name=name,
                                         prefs=prefs,
                                         context=context.__class__.__name__)

        # INSTANTIATE PROJECTIONS SPECIFIED IN projections ARG OR params[PROJECTIONS:<>]
        if PROJECTIONS in self.paramsCurrent and self.paramsCurrent[PROJECTIONS]:
            self._instantiate_projections(self.paramsCurrent[PROJECTIONS], context=context)
        else:
            # No projections specified, so none will be created here
            # IMPLEMENTATION NOTE:  This is where a default projection would be implemented
            #                       if params = NotImplemented or there is no param[PROJECTIONS]
            pass

    def _handle_size(self, size, variable):
        """Overwrites the parent method in Component.py, because the variable of a State
            is generally 1D, rather than 2D as in the case of Mechanisms"""
        if size is not NotImplemented:

            def checkAndCastInt(x):
                if not isinstance(x, numbers.Number):
                    raise StateError("Size ({}) is not a number.".format(x))
                if x < 1:
                    raise StateError("Size ({}) is not a positive number.".format(x))
                try:
                    int_x = int(x)
                except:
                    raise StateError(
                        "Failed to convert size argument ({}) for {} {} to an integer. For States, size "
                        "should be a number, which is an integer or can be converted to integer.".
                        format(x, type(self), self.name))
                if int_x != x:
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("When size ({}) was cast to integer, its value changed to {}.".format(x, int_x))
                return int_x

            # region Convert variable to a 1D array, cast size to an integer
            if size is not None:
                size = checkAndCastInt(size)
            try:
                if variable is not None:
                    variable = self._update_variable(np.atleast_1d(variable))
            except:
                raise StateError("Failed to convert variable (of type {}) to a 1D array.".format(type(variable)))
            # endregion

            # region if variable is None and size is not None, make variable a 1D array of zeros of length = size
            if variable is None and size is not None:
                try:
                    variable = self._update_variable(np.zeros(size))
                except:
                    raise ComponentError("variable (perhaps default_variable) was not specified, but PsyNeuLink "
                                         "was unable to infer variable from the size argument, {}. size should be"
                                         " an integer or able to be converted to an integer. Either size or "
                                         "variable must be specified.".format(size))
            #endregion

            if variable is not None and size is not None:  # try tossing this "if" check
                # If they conflict, raise exception
                if size != len(variable):
                    raise StateError("The size arg of {} ({}) conflicts with the length of its variable arg ({})".
                                     format(self.name, size, variable))

        return variable

    def _validate_variable(self, variable, context=None):
        """Validate variable and return validated variable

        Sets self.base_value = self.value = variable
        Insures that it is a number of list or tuple of numbers

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True
        """

        variable = self._update_variable(super(State, self)._validate_variable(variable, context))

        if not context:
            context = kwAssign + ' Base Value'
        else:
            context = context + kwAssign + ' Base Value'

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """validate projection specification(s)

        Call super (Component._validate_params()
        Validate following params:
            + PROJECTIONS:  <entry or list of entries>; each entry must be one of the following:
                + Projection object
                + Projection class
                + specification dict, with the following entries:
                    + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                    + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            # IMPLEMENTATION NOTE: TBI - When learning projection is implemented
            # + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
            #     ParameterState, projection, 2-item tuple or value
        """

        if PROJECTIONS in request_set and request_set[PROJECTIONS]:
            # if projection specification is an object or class reference, needs to be wrapped in a list
            # - to be consistent with paramClassDefaults
            # - for consistency of treatment below
            projections = request_set[PROJECTIONS]
            if not isinstance(projections, list):
                projections = [projections]
                request_set[PROJECTIONS] = projections
        else:
            # If no projections, ignore (none will be created)
            projections = None

        super(State, self)._validate_params(request_set, target_set, context=context)

        if projections:
            # Validate projection specs in list
            from PsyNeuLink.Components.Projections.Projection import Projection
            for projection in projections:
                try:
                    issubclass(projection, Projection)
                except TypeError:
                    if (isinstance(projection, Projection) or iscompatible(projection, dict)):
                        continue
                    else:
                        if self.prefs.verbosePref:
                            print("{0} in {1} is not a projection, projection type, or specification dict; "
                                  "{2} will be used to create default {3} for {4}".
                                format(projection,
                                       self.__class__.__name__,
                                       target_set[PROJECTION_TYPE],
                                       self.owner.name))

    def _instantiate_function(self, context=None):
        """Insure that output of function (self.value) is compatible with its input (self.instance_defaults.variable)

        This constraint reflects the role of State functions:
            they simply update the value of the State;
            accordingly, their variable and value must be compatible
        """

        var_is_matrix = False
        # If variable is a matrix (e.g., for the MATRIX ParameterState of a MappingProjection),
        #     it needs to be embedded in a list so that it is properly handled by LinearCombination
        #     (i.e., solo matrix is returned intact, rather than treated as arrays to be combined);
        # Notes:
        #     * this is not a problem when LinearCombination is called in State.update(), since that puts
        #         projection values in a list before calling LinearCombination to combine them
        #     * it is removed from the list below, after calling _instantiate_function
        # FIX: UPDATE WITH MODULATION_MODS REMOVE THE FOLLOWING COMMENT:
        #     * no change is made to PARAMETER_MODULATION_FUNCTION here (matrices may be multiplied or added)
        #         (that is handled by the individual State subclasses (e.g., ADD is enforced for MATRIX ParameterState)
        if (
            (
                (inspect.isclass(self.function) and issubclass(self.function, LinearCombination))
                or isinstance(self.function, LinearCombination)
            )
            and (
                isinstance(self.instance_defaults.variable, np.matrix)
                or (
                    isinstance(self.instance_defaults.variable, np.ndarray)
                    and self.instance_defaults.variable.ndim >= 2
                )
            )
        ):
            self.instance_defaults.variable = [self.instance_defaults.variable]
            var_is_matrix = True

        super()._instantiate_function(context=context)

        # If it is a matrix, remove from list in which it was embedded after instantiating and evaluating function
        if var_is_matrix:
            self.instance_defaults.variable = self.instance_defaults.variable[0]

        # Ensure that output of the function (self.value) is compatible with (same format as) its input (self.instance_defaults.variable)
        #     (this enforces constraint that State functions should only combine values from multiple projections,
        #     but not transform them in any other way;  so the format of its value should be the same as its variable).
        if not iscompatible(self.instance_defaults.variable, self.value):
            raise StateError(
                "Output ({0}: {1}) of function ({2}) for {3} {4} of {5}"
                " must be the same format as its input ({6}: {7})".format(
                    type(self.value).__name__,
                    self.value,
                    self.function.__self__.componentName,
                    self.name,
                    self.__class__.__name__,
                    self.owner.name,
                    self.instance_defaults.variable.__class__.__name__,
                    self.instance_defaults.variable
                )
            )

    def _instantiate_projections(self, projections, context=None):
        """Implement any Projection(s) to/from State specified in PROJECTIONS entry of params arg

        Must be implemented by subclasss, to handle interpretation of projection specification(s)
        in a class-appropriate manner:
            PathwayProjections:
              InputState: _instantiate_projections_to_state (.pathway_afferents)
              ParameterState: disallowed
              OutputState: _instantiate_projections_from_state (.efferents)
              ModulatorySignal: disallowed
            ModulatoryProjections:
              InputState, OutputState and ParameterState:  _instantiate_projections_to_state (mod_afferents)
              ModulatorySignal: _instantiate_projections_from_state (.efferents)
        """

        raise StateError("{} must implement _instantiate_projections (called for {})".
                         format(self.__class__.__name__,
                                self.name))

    def _instantiate_projections_to_state(self, projections, context=None):
        """Instantiate projections to a State and assign them to self.path_afferents

        For each spec in projections arg, check that it is one or a list of any of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that receiver is self
            checks that projection function output is compatible with self.value
        + State object or State class
            check that it is compatible with (i.e., a legitimate sender for) projection
            if it is class, instantiate default
            assign as sender of the projection
        + Mechanism object:
            check that it is compatible with (i.e., a legitimate sender for) projection
        + specification dict (usually from PROJECTIONS entry of params dict):
            checks that projection function output is compatible with self.value
            implements projection
            dict must contain:
                + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[PROJECTION_TYPE]
        For each projection:
            if it is a MappingProjection, it is added to self.path_afferents
            if it is a LearningProjection, ControlProjection, or GatingProjection, it is added to self.mod_afferents
        If kwMStateProjections is absent or empty, no projections are created
        """

        from PsyNeuLink.Components.Projections.Projection import Projection_Base
        from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection \
            import PathwayProjection_Base
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection \
            import ModulatoryProjection_Base
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base

        # If specification is not a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projections
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        state_name_string = self.name
        item_prefix_string = ""
        item_suffix_string = state_name_string + " ({} for {})".format(self.__class__.__name__, self.owner.name,)
        default_string = ""
        kwDefault = "default "

        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]

        # Instantiate each projection specification in the projection_list, and
        # - insure it is in self.path_afferents
        # - insure the output of its function is compatible with self.value
        for projection_spec in projection_list:

            # If there is more than one projection specified, construct messages for use in case of failure
            if len(projection_list) > 1:
                item_prefix_string = "Item {0} of projection list for {1}: ".\
                    format(projection_list.index(projection_spec)+1, state_name_string)
                item_suffix_string = ""

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION_SPEC
# FIX: OR A _parse_projection_spec METHOD
            projection_object = None # flags whether projection object has been instantiated; doesn't store object
            projection_type = None   # stores type of projection to instantiate
            sender = None
            projection_params = {}

            # PARSE AND INSTANTIATE PROJECTION_SPEC --------------------------------------------------------------------

            # If projection_spec is a Projection object:
            # - call _check_projection_receiver() to check that receiver is self; if not, it:
            #     returns object with receiver reassigned to self if chosen by user
            #     else, returns new (default) PROJECTION_TYPE object with self as receiver
            #     note: in that case, projection will be in self.path_afferents list
            if isinstance(projection_spec, Projection_Base):
                if projection_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:
                    if isinstance(projection_spec, ModulatoryProjection_Base):
                        # Assign projection to mod_afferents
                        self.mod_afferents.append(projection_spec)
                        projection_spec.init_args[RECEIVER] = self
                        # Skip any further initialization for now
                        #   (remainder will occur as part of deferred init for
                        #    ControlProjection or GatingProjection)
                        continue

                    # Complete init for other (presumably Mapping) projections
                    else:
                        # Assume init was deferred because receiver could not be determined previously
                        #  (e.g., specified in function arg for receiver object, or as standalone projection in script)
                        # Assign receiver to init_args and call _deferred_init for projection
                        projection_spec.init_args[RECEIVER] = self
                        projection_spec.init_args['name'] = self.owner.name+' '+self.name+' '+projection_spec.className
                        # FIX: REINSTATE:
                        # projection_spec.init_args['context'] = context
                        projection_spec._deferred_init()
                projection_object, default_class_name = self._check_projection_receiver(
                                                                                    projection_spec=projection_spec,
                                                                                    messages=[item_prefix_string,
                                                                                              item_suffix_string,
                                                                                              state_name_string],
                                                                                    context=self)
                # If projection's name has not been assigned, base it on State's name:
                if default_class_name:
                    # projection_object.name = projection_object.name.replace(default_class_name, self.name)
                    projection_object.name = self.name + '_' + projection_object.name
                    # Used for error message
                    default_string = kwDefault
# FIX:  REPLACE DEFAULT NAME (RETURNED AS DEFAULT) PROJECTION_SPEC NAME WITH State'S NAME, LEAVING INDEXED SUFFIX INTACT

            # If projection_spec is a State or State class
            # - check that it is appropriate for the type of projection
            # - create default instance if it is a class (it will use deferred_init since owner is not yet known)
            # - Assign to sender (for assignment as projection's sender below)
            # - default projection itself will be created below
            elif (isinstance(projection_spec, State) or
                          inspect.isclass(projection_spec) and issubclass(projection_spec, State)):
                # If it is State, get its type (for check below)
                if isinstance(projection_spec, State):
                    state_type = type(projection_spec)
                # If it is State class, instantiate default
                else:
                    projection_spec = projection_spec()
                # Check appropriateness of State
                _check_projection_sender_compatability(self, default_projection_type, state_type)
                # Assign State as projections's sender (for use below)
                sender = projection_spec

            # If projection_spec is a Mechanism:
            # - check compatibility with projection's type
            # - default projection itself will be created below
            elif isinstance(projection_spec, Mechanism):
                _check_projection_sender_compatability(self, default_projection_type, type(projection_spec))
                # If Mechanism is a ProcessingMechanism, assign its primary OutputState as the sender
                # (for ModulatoryProjections, don't assign sender, which will defer initialization)
                # from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism \
                #     import ProcessingMechanism_Base
                if isinstance(projection_spec, ProcessingMechanism_Base):
                    sender = projection_spec.output_state

            # If projection_spec is a dict:
            # - get projection_type
            # - get projection_params
            # Note: this gets projection_type but does NOT not instantiate projection; so,
            #       projection is NOT yet in self.path_afferents list
            elif isinstance(projection_spec, dict):
                # Get projection type from specification dict
                try:
                    projection_type = projection_spec[PROJECTION_TYPE]
                    _check_projection_sender_compatability(self, default_projection_type, projection_type)
                except KeyError:
                    projection_type = default_projection_type
                    default_string = kwDefault
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_TYPE,
                                     PROJECTIONS,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))
                else:
                    # IMPLEMENTATION NOTE:  can add more informative reporting here about reason for failure
                    projection_type, error_str = self._parse_projection_ref(projection_spec=projection_type,
                                                                           context=self)
                    if error_str and self.prefs.verbosePref:
                        warnings.warn("{0}{1} {2}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_TYPE,
                                     error_str,
                                     PROJECTIONS,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

                # Get projection params from specification dict
                try:
                    projection_params = projection_spec[PROJECTION_PARAMS]
                except KeyError:
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_PARAMS,
                                     PROJECTIONS, state_name_string,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

            # Check if projection_spec is class ref or keyword string constant for one
            # Note: this gets projection_type but does NOT instantiate the projection (that happens below),
            #       so projection is NOT yet in self.path_afferents list
            else:
                projection_type, err_str = self._parse_projection_ref(projection_spec=projection_spec,context=self)
                if err_str and self.verbosePref:
                    warnings.warn("{0}{1} {2}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 err_str,
                                 PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

            # If neither projection_object nor projection_type have been assigned, assign default type
            # Note: this gets projection_type but does NOT instantiate projection; so,
            #       projection is NOT yet in self.path_afferents list
            if not projection_object and not projection_type:
                    projection_type = default_projection_type
                    default_string = kwDefault
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} is not a Projection object or specification for one{2}; "
                              "default {3} will be assigned".
                              format(item_prefix_string,
                                     projection_spec.name,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

            # If projection_object has not been assigned, instantiate projection_type
            # Note: this automatically assigns projection to self.path_afferents and to it's sender's efferents list;
            #       when a projection is instantiated, it assigns itself to:
            #           its receiver's .path_afferents attribute (in Projection._instantiate_receiver)
            #           its sender's .efferents attribute (in Projection._instantiate_sender)
            if not projection_object:
                kwargs = {RECEIVER:self,
                          NAME:self.owner.name+' '+self.name+' '+projection_type.className,
                          PARAMS:projection_params,
                          CONTEXT:context}
                # If the projection_spec was a State (see above) and assigned as the sender, assign to SENDER arg
                if sender:
                    kwargs.update({SENDER:sender})
                # If the projection was specified with a keyword or attribute value
                #     then move it to the relevant entry of the params dict for the projection
                # If projection_spec was in the form of a matrix keyword, move it to a matrix entry in the params dict
                if issubclass(projection_type, PathwayProjection_Base) and projection_spec in MATRIX_KEYWORD_SET:
                    kwargs.update({MATRIX:projection_spec})
                # If projection_spec was in the form of a ModulationParam value,
                #    move it to a MODULATION entry in the params dict
                elif (issubclass(projection_type, ModulatoryProjection_Base) and
                          isinstance(projection_spec, ModulationParam)):
                    kwargs[PARAMS].update({MODULATION:projection_spec})
                projection_spec = projection_type(**kwargs)

            # Check that output of projection's function (projection_spec.value is compatible with
            #    variable of the State to which it projects;  if it is not, raise exception:
            # The buck stops here; can't modify projection's function to accommodate the State,
            #    or there would be an unmanageable regress of reassigning projections,
            #    requiring reassignment or modification of sender OutputStates, etc.

            # Initialization of projection is deferred
            if projection_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:
                # Assign instantiated "stub" so it is found on deferred initialization pass (see Process)
                if isinstance(projection_spec, ModulatoryProjection_Base):
                    self.mod_afferents.append(projection_spec)
                else:
                    self.path_afferents.append(projection_spec)
                continue

            # Projection was instantiated, so:
            #    - validate value
            #    - assign to State's path_afferents or mod_afferents list
            # If it is a ModualatoryProjection:
            #    - check that projection's value is compatible with value of the function param being modulated
            #    - assign projection to mod_afferents
            if isinstance(projection_spec, ModulatoryProjection_Base):
                function_param_value = _get_modulated_param(self, projection_spec).function_param_val
                # Match the projection's value with the value of the function parameter
                mod_proj_spec_value = type_match(projection_spec.value, type(function_param_value))
                # If the match was successful (i.e., they are compatible), assign the projection to mod_afferents
                if function_param_value is None or iscompatible(function_param_value, mod_proj_spec_value):
                    # Avoid duplicates, since instantiation of projection (e.g, by Mechanism)
                    #    may have already called this method and assigned projection to self.mod_afferents
                    if not projection_spec in self.mod_afferents:
                        self.mod_afferents.append(projection_spec)
                    continue
            # Otherwise:
            #    - check that projection's value is compatible with the State's variable
            #    - assign projection to path_afferents
            else:
                if iscompatible(self.instance_defaults.variable, projection_spec.value):
                    # This is needed to avoid duplicates, since instantiation of projection (e.g., of ControlProjection)
                    #    may have already called this method and assigned projection to self.path_afferents list
                    if not projection_spec in self.path_afferents:
                        self.path_afferents.append(projection_spec)
                    continue

            # Projection specification is not valid
            raise StateError("{}Output of function for {}{} ( ({})) is not compatible with value of {} ({})".
                             format(item_prefix_string,
                                    default_string,
                                    projection_spec.name,
                                    projection_spec.value,
                                    item_suffix_string,
                                    self.value))

    def _instantiate_projection_from_state(self, projection_spec, receiver, context=None):
        """Instantiate outgoing projection from a State and assign it to self.efferents

        Check that projection_spec is one of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that sender is self
            checks that self.value is compatible with projection's function variable
        + specification dict:
            checks that self.value is compatiable with projection's function variable
            implements projection
            dict must contain:
                + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[PROJECTION_TYPE]
        Projection is added to self.efferents
        If kwMStateProjections is absent or empty, no projections are created
        """

        from PsyNeuLink.Components.Projections.Projection import Projection_Base, ProjectionRegistry

        state_name_string = self.name
        item_prefix_string = ""
        item_suffix_string = state_name_string + " ({} for {})".format(self.__class__.__name__, self.owner.name,)
        default_string = ""
        kwDefault = "default "

        default_projection_type = ProjectionRegistry[self.paramClassDefaults[PROJECTION_TYPE]].subclass

        # Instantiate projection specification and
        # - insure it is in self.efferents
        # - insure self.value is compatible with the projection's function variable

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION_SPEC
        projection_object = None # flags whether projection object has been instantiated; doesn't store object
        projection_type = None   # stores type of projection to instantiate
        projection_params = {}

        # VALIDATE RECEIVER
        # # MODIFIED 7/8/17 OLD:
        # # Must be an InputState or ParameterState
        # from PsyNeuLink.Components.States.InputState import InputState
        # from PsyNeuLink.Components.States.ParameterState import ParameterState
        # if not isinstance(receiver, (InputState, ParameterState, Mechanism)):
        #     raise StateError("Receiver {} of {} from {} must be an InputState, ParameterState".
        #                      format(receiver, projection_spec, self.name))
        # MODIFIED 7/8/17 NEW:
        # ALLOW SPEC TO BE ANY STATE (INCLUDING OutPutState, FOR GATING PROJECTIONS)
        # OR MECHANISM (IN WHICH CASE PRIMARY INPUTSTATE IS ASSUMED)
        # Must be an InputState or ParameterState
        from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
        from PsyNeuLink.Components.States.State import State
        if not isinstance(receiver, (State, Mechanism)):
            raise StateError("Receiver ({}) of {} from {} must be a State or Mechanism".
                             format(receiver, projection_spec, self.name))
        # If receiver is a Mechanism, assume use of primary InputState (and warn if verbose is set)
        if isinstance(receiver, Mechanism):
            if self.verbosePref:
                warnings.warn("Receiver {} of {} from {} is a Mechanism, so its primary InputState will be used".
                              format(receiver, projection_spec, self.name))
            receiver = receiver.input_state

        # MODIFIED 7/8/17 END

        # INSTANTIATE PROJECTION_SPEC
        # If projection_spec is a Projection object:
        # - call _check_projection_sender() to check that sender is self; if not, it:
        #     returns object with sender reassigned to self if chosen by user
        #     else, returns new (default) PROJECTION_TYPE object with self as sender
        #     note: in that case, projection will be in self.efferents list
        if isinstance(projection_spec, Projection_Base):
            projection_object, default_class_name = self._check_projection_sender(projection_spec=projection_spec,
                                                                                 receiver=receiver,
                                                                                 messages=[item_prefix_string,
                                                                                           item_suffix_string,
                                                                                           state_name_string],
                                                                                 context=self)
            # If projection's name has not been assigned, base it on State's name:
            if default_class_name:
                # projection_object.name = projection_object.name.replace(default_class_name, self.name)
                projection_object.name = self.name + '_' + projection_object.name
                # Used for error message
                default_string = kwDefault
# FIX:  REPLACE DEFAULT NAME (RETURNED AS DEFAULT) PROJECTION_SPEC NAME WITH State'S NAME, LEAVING INDEXED SUFFIX INTACT

        # If projection_spec is a dict:
        # - get projection_type
        # - get projection_params
        # Note: this gets projection_type but does NOT not instantiate projection; so,
        #       projection is NOT yet in self.efferents list
        elif isinstance(projection_spec, dict):
            # Get projection type from specification dict
            try:
                projection_type = projection_spec[PROJECTION_TYPE]
            except KeyError:
                projection_type = default_projection_type
                default_string = kwDefault
                if self.prefs.verbosePref:
                    print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))
            else:
                # IMPLEMENTATION NOTE:  can add more informative reporting here about reason for failure
                projection_type, error_str = self._parse_projection_ref(projection_spec=projection_type,
                                                                       context=self)
                if error_str:
                    print("{0}{1} {2}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 error_str,
                                 PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

            # Get projection params from specification dict
            try:
                projection_params = projection_spec[PROJECTION_PARAMS]
            except KeyError:
                if self.prefs.verbosePref:
                    print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_PARAMS,
                                 PROJECTIONS, state_name_string,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

        # Check if projection_spec is class ref or keyword string constant for one
        # Note: this gets projection_type but does NOT instantiate the projection,
        #       so projection is NOT yet in self.efferents list
        else:
            projection_type, err_str = self._parse_projection_ref(projection_spec=projection_spec,context=self)
            if err_str:
                print("{0}{1} {2}; default {4} will be assigned".
                      format(item_prefix_string,
                             PROJECTION_TYPE,
                             err_str,
                             PROJECTIONS,
                             item_suffix_string,
                             default_projection_type.__class__.__name__))

        # If neither projection_object nor projection_type have been assigned, assign default type
        # Note: this gets projection_type but does NOT not instantiate projection; so,
        #       projection is NOT yet in self.path_afferents list
        if not projection_object and not projection_type:
                projection_type = default_projection_type
                default_string = kwDefault
                if self.prefs.verbosePref:
                    print("{0}{1} is not a Projection object or specification for one{2}; "
                          "default {3} will be assigned".
                          format(item_prefix_string,
                                 projection_spec.name,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

        # If projection_object has not been assigned, instantiate projection_type
        # Note: this automatically assigns projection to self.efferents and
        #       to it's receiver's afferents list:
        #           when a projection is instantiated, it assigns itself to:
        #               MODIFIED 7/8/17: QUESTION: DOES THE FOLLOWING FAIL TO MENTION .mod_afferents for Mod Projs?
        #               its receiver's .path_afferents attribute (in Projection._instantiate_receiver)
        #               its sender's .efferents list attribute (in Projection._instantiate_sender)
        if not projection_object:
            projection_spec = projection_type(sender=self,
                                              receiver=receiver,
                                              name=self.name+'_'+projection_type.className,
                                              params=projection_params,
                                              context=context)

        # Check that self.value is compatible with projection's function variable
        if not iscompatible(self.value, projection_spec.instance_defaults.variable):
            raise StateError("{0}Output ({1}) of {2} is not compatible with variable ({3}) of function for {4}".
                  format(
                         item_prefix_string,
                         self.value,
                         item_suffix_string,
                         projection_spec.instance_defaults.variable,
                         projection_spec.name
                         ))

        # If projection is valid, assign to State's efferents list
        else:
            # Check for duplicates (since instantiation of projection may have already called this method
            #    and assigned projection to self.efferents list)
            if not projection_spec in self.efferents:
                self.efferents.append(projection_spec)

    def _check_projection_receiver(self, projection_spec, messages=None, context=None):
        """Check whether Projection object references State as receiver and, if not, return default Projection object

        Arguments:
        - projection_spec (Projection object)
        - message (list): list of three strings - prefix and suffix for error/warning message, and State name
        - context (object): ref to State object; used to identify PROJECTION_TYPE and name

        Returns: tuple (Projection object, str); second value is name of default projection, else None

        :param self:
        :param projection_spec: (Projection object)
        :param messages: (list)
        :param context: (State object)
        :return: (tuple) Projection object, str) - second value is false if default was returned
        """

        prefix = 0
        suffix = 1
        name = 2
        if messages is None:
            messages = ["","","",context.__class__.__name__]
        message = "{}{} is a projection of the correct type for {}, but its receiver is not assigned to {}." \
                  " \nReassign (r) or use default projection (d)?:".format(messages[prefix],
                                                                           projection_spec.name,
                                                                           projection_spec.receiver.name,
                                                                           messages[suffix])

        if projection_spec.receiver is not self:
            reassign = input(message)
            while reassign != 'r' and reassign != 'd':
                reassign = input("Reassign {0} to {1} or use default (r/d)?:".
                                 format(projection_spec.name, messages[name]))
            # User chose to reassign, so return projection object with State as its receiver
            if reassign == 'r':
                projection_spec.receiver = self
                # IMPLEMENTATION NOTE: allow the following, since it is being carried out by State itself
                self.path_afferents.append(projection_spec)
                if self.prefs.verbosePref:
                    print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
                return (projection_spec, None)
            # User chose to assign default, so return default projection object
            elif reassign == 'd':
                print("Default {0} will be used for {1}".
                      format(projection_spec.name, messages[name]))
                return (self.paramsCurrent[PROJECTION_TYPE](receiver=self),
                        self.paramsCurrent[PROJECTION_TYPE].className)
                #     print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
            else:
                raise StateError("Program error:  reassign should be r or d")

        return (projection_spec, None)

    def _check_projection_sender(self, projection_spec, receiver, messages=None, context=None):
        """Check whether Projection object references State as sender and, if not, return default Projection object

        Arguments:
        - projection_spec (Projection object)
        - message (list): list of three strings - prefix and suffix for error/warning message, and State name
        - context (object): ref to State object; used to identify PROJECTION_TYPE and name

        Returns: tuple (Projection object, str); second value is name of default projection, else None

        :param self:
        :param projection_spec: (Projection object)
        :param messages: (list)
        :param context: (State object)
        :return: (tuple) Projection object, str) - second value is false if default was returned
        """

        prefix = 0
        suffix = 1
        name = 2
        if messages is None:
            messages = ["","","",context.__class__.__name__]
        #FIX: NEED TO GET projection_spec.name VS .__name__ STRAIGHT BELOW
        message = "{}{} is a projection of the correct type for {}, but its sender is not assigned to {}." \
                  " \nReassign (r) or use default projection(d)?:".format(messages[prefix],
                                                                          projection_spec.name,
                                                                          projection_spec.sender,
                                                                          messages[suffix])

        if not projection_spec.sender is self:
            reassign = input(message)
            while reassign != 'r' and reassign != 'd':
                reassign = input("Reassign {0} to {1} or use default (r/d)?:".
                                 format(projection_spec.name, messages[name]))
            # User chose to reassign, so return projection object with State as its sender
            if reassign == 'r':
                projection_spec.sender = self
                # IMPLEMENTATION NOTE: allow the following, since it is being carried out by State itself
                self.efferents.append(projection_spec)
                if self.prefs.verbosePref:
                    print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
                return (projection_spec, None)
            # User chose to assign default, so return default projection object
            elif reassign == 'd':
                print("Default {0} will be used for {1}".
                      format(projection_spec.name, messages[name]))
                return (self.paramsCurrent[PROJECTION_TYPE](sender=self, receiver=receiver),
                        self.paramsCurrent[PROJECTION_TYPE].className)
                #     print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
            else:
                raise StateError("Program error:  reassign should be r or d")

        return (projection_spec, None)

    def _parse_projection_ref(self,
                             projection_spec,
                             # messages=NotImplemented,
                             context=None):
        """Take projection ref and return ref to corresponding type or, if invalid, to  default for context

        Arguments:
        - projection_spec (Projection subclass or str):  str must be a keyword constant for a Projection subclass
                                                         or an alias for one:  LEARNING = LEARNING_PROJECTION
                                                                               CONTROL = CONTROL_PROJECTION
                                                                               GATING = GATING_PROJECTION
        - context (str):

        Returns tuple: (Projection subclass or None, error string)

        :param projection_spec: (Projection subclass or str)
        :param messages: (list)
        :param context: (State object)
        :return: (Projection subclass, string)
        """
        try:
            # Try projection spec as class ref
            is_projection_class = issubclass(projection_spec, Projection)
        except TypeError:
            # Try projection spec as keyword string constant
            if isinstance(projection_spec, str):

                # de-alias convenience keywords:
                if projection_spec is LEARNING:
                    projection_spec = LEARNING_PROJECTION
                if projection_spec is CONTROL:
                    projection_spec = CONTROL_PROJECTION
                if projection_spec is GATING:
                    projection_spec = GATING_PROJECTION

                try:
                    from PsyNeuLink.Components.Projections.Projection import ProjectionRegistry
                    projection_spec = ProjectionRegistry[projection_spec].subclass
                except KeyError:
                    # projection_spec was not a recognized key
                    return (None, "not found in ProjectionRegistry")
                # projection_spec was legitimate keyword
                else:
                    return (projection_spec, None)
            # projection_spec was neither a class reference nor a keyword
            else:
                return (None, "neither a class reference nor a keyword")
        else:
            # projection_spec was a legitimate class
            if is_projection_class:
                return (projection_spec, None)
            # projection_spec was class but not Projection
            else:
                return (None, "not a Projection subclass")#


    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Update each projection, combine them, and assign return result

        Call update for each projection in self.path_afferents (passing specified params)
        Note: only update LearningSignals if context == LEARNING; otherwise, just get their value
        Call self.function (default: LinearCombination function) to combine their values
        Returns combined values of

    Arguments:
    - context (str)

    :param context: (str)
    :return: None

    """

        # region SET UP ------------------------------------------------------------------------------------------------
        # Get State-specific param_specs
        try:
            # Get State params
            self.stateParams = params[self.paramsType]
        except (KeyError, TypeError):
            self.stateParams = {}
        except (AttributeError):
            raise StateError("PROGRAM ERROR: paramsType not specified for {}".format(self.name))
        #endregion

        # Flag format of input
        if isinstance(self.value, numbers.Number):
            # Treat as single real value
            value_is_number = True
        else:
            # Treat as vector (list or np.array)
            value_is_number = False

        # region AGGREGATE INPUT FROM PROJECTIONS -----------------------------------------------------------------------------

        # Get type-specific params from PROJECTION_PARAMS
        mapping_params = merge_param_dicts(self.stateParams, MAPPING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        learning_projection_params = merge_param_dicts(self.stateParams, LEARNING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        control_projection_params = merge_param_dicts(self.stateParams, CONTROL_PROJECTION_PARAMS, PROJECTION_PARAMS)
        gating_projection_params = merge_param_dicts(self.stateParams, GATING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        #endregion

        #For each projection: get its params, pass them to it, get the projection's value, and append to relevant list
        self._path_proj_values = []
        for value in self._mod_proj_values:
            self._mod_proj_values[value] = []

        from PsyNeuLink.Components.Process import ProcessInputState
        from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection \
            import PathwayProjection_Base
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection \
            import ModulatoryProjection_Base
        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

        # If owner is a Mechanism, get its execution_id
        if isinstance(self.owner, (Mechanism, Process)):
            self_id = self.owner._execution_id
        # If owner is a MappingProjection, get it's sender's execution_id
        elif isinstance(self.owner, MappingProjection):
            self_id = self.owner.sender.owner._execution_id
        else:
            raise StateError("PROGRAM ERROR: Object ({}) of type {} has a {}, but this is only allowed for "
                             "Mechanisms and MappingProjections".
                             format(self.owner.name, self.owner.__class__.__name__, self.__class__.__name__,))


        modulatory_override = False

        # Get values of all Projections
        for projection in self.all_afferents:

            # Only update if sender has also executed in this round
            #     (i.e., has same execution_id as owner)
            # Get sender's execution id
            if hasattr(projection, 'sender'):
                sender = projection.sender
            else:
                if self.verbosePref:
                    warnings.warn("{} to {} {} of {} ignored [has no sender]".format(projection.__class__.__name__,
                                                                                     self.name,
                                                                                     self.__class__.__name__,
                                                                                     self.owner.name))
                continue

            sender_id = sender.owner._execution_id
            if isinstance(sender.owner, Mechanism):
                if (not sender.owner.ignore_execution_id) and (sender_id != self_id):
                    continue
            else:
                if sender_id != self_id:
                    continue

            # Only accept projections from a Process to which the owner Mechanism belongs
            if isinstance(sender, ProcessInputState):
                if not sender.owner in self.owner.processes.keys():
                    continue

            # Merge with relevant projection type-specific params
            if isinstance(projection, MappingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, mapping_params, )
            elif isinstance(projection, LearningProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, learning_projection_params)
            elif isinstance(projection, ControlProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, control_projection_params)
            elif isinstance(projection, GatingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, gating_projection_params)
            if not projection_params:
                projection_params = None

            # FIX: UPDATE FOR LEARNING
            # Update LearningSignals only if context == LEARNING;  otherwise, assign zero for projection_value
            # Note: done here rather than in its own method in order to exploit parsing of params above
            if isinstance(projection, LearningProjection) and not LEARNING in context:
                # projection_value = projection.value
                projection_value = projection.value * 0.0
            else:
                projection_value = projection.execute(params=projection_params,
                                                      time_scale=time_scale,
                                                      context=context)

            # If this is initialization run and projection initialization has been deferred, pass
            if INITIALIZING in context and projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                continue

            if isinstance(projection, PathwayProjection_Base):
                # Add projection_value to list of TransmissiveProjection values (for aggregation below)
                self._path_proj_values.append(projection_value)

            # If it is a ModulatoryProjection, add its value to the list in the dict entry for the relevant mod_param
            elif isinstance(projection, ModulatoryProjection_Base):
                # Get the meta_param to be modulated from modulation attribute of the  projection's ModulatorySignal
                #    and get the function parameter to be modulated to type_match the projection value below
                mod_meta_param, mod_param_name, mod_param_value = _get_modulated_param(self, projection)
                # If meta_param is DISABLE, ignore the ModulatoryProjection
                if mod_meta_param is Modulation.DISABLE:
                    continue
                if mod_meta_param is Modulation.OVERRIDE:
                    # If paramValidationPref is set, allow all projections to be processed
                    #    to be sure there are no other conflicting OVERRIDES assigned
                    if self.owner.paramValidationPref:
                        if modulatory_override:
                            raise StateError("Illegal assignment of {} to more than one {} ({} and {})".
                                             format(MODULATION_OVERRIDE, MODULATORY_SIGNAL,
                                                    projection.name, modulatory_override[2]))
                        modulatory_override = (MODULATION_OVERRIDE, projection_value, projection)
                        continue
                    # Otherwise, for efficiency, assign OVERRIDE value to State here and return
                    else:
                        self.value = type_match(projection_value, type(self.value))
                        return
                else:
                    mod_value = type_match(projection_value, type(mod_param_value))
                self._mod_proj_values[mod_meta_param].append(mod_value)

        # Handle ModulatoryProjection OVERRIDE
        #    if there is one and it wasn't been handled above (i.e., if paramValidation is set)
        if modulatory_override:
            self.value = type_match(modulatory_override[1], type(self.value))
            return

        # AGGREGATE ModulatoryProjection VALUES  -----------------------------------------------------------------------

        # For each modulated parameter of the State's function,
        #    combine any values received from the relevant projections into a single modulation value
        #    and assign that to the relevant entry in the params dict for the State's function.
        for mod_param, value_list in self._mod_proj_values.items():
            if value_list:
                aggregated_mod_val = mod_param.reduce(value_list)
                function_param = self.function_object.params[mod_param.attrib_name]
                if not FUNCTION_PARAMS in self.stateParams:
                    self.stateParams[FUNCTION_PARAMS] = {function_param: aggregated_mod_val}
                else:
                    self.stateParams[FUNCTION_PARAMS].update({function_param: aggregated_mod_val})

        # CALL STATE'S function TO GET ITS VALUE  ----------------------------------------------------------------------
        try:
            # pass only function params (which implement the effects of any ModulatoryProjections)
            function_params = self.stateParams[FUNCTION_PARAMS]
        except (KeyError, TypeError):
            function_params = None
        self.value = self._execute(function_params=function_params, context=context)

    def execute(self, input=None, time_scale=None, params=None, context=None):
        return self.function(variable=input, params=params, time_scale=time_scale, context=context)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, assignment):
        self._owner = assignment

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):

        # MODIFIED 7/8/17 OLD:
        # from math import isnan
        # if isinstance(assignment, np.ndarray) and assignment.ndim == 2 and isnan(assignment[0][0]):
        #             TEST = True
        # MODIFIED 7/8/17 END

        self._value = assignment

        # Store value in log if specified
        # Get logPref
        if self.prefs:
            log_pref = self.prefs.logPref

        # Get context
        try:
            curr_frame = inspect.currentframe()
            prev_frame = inspect.getouterframes(curr_frame, 2)
            context = inspect.getargvalues(prev_frame[1][0]).locals['context']
        except KeyError:
            context = ""

        # If context is consistent with log_pref, record value to log
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context and kwAssign in context))):
            self.owner.log.entries[self.name] = LogEntry(CurrentTime(), context, assignment)
            # self.owner.log.entries[self.name] = LogEntry(CentralClock, context, assignment)

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, assignment):
        self._projections = assignment

    @property
    def all_afferents(self):
        return self.path_afferents + self.mod_afferents


def _instantiate_state_list(owner,
                           state_list,              # list of State specs, (state_spec, params) tuples, or None
                           state_type,              # StateType subclass
                           state_param_identifier,  # used to specify state_type State(s) in params[]
                           constraint_value,       # value(s) used as default for State and to check compatibility
                           constraint_value_name,  # name of constraint_value type (e.g. variable, output...)
                           context=None):
    """Instantiate and return a ContentAddressableList of States specified in state_list

    Arguments:
    - state_type (class): State class to be instantiated
    - state_list (list): List of State specifications (generally from owner.paramsCurrent[kw<State>]),
                             each item of which must be a:
                                 string (used as name)
                                 value (used as constraint value)
                                 # ??CORRECT: (state_spec, params_dict) tuple
                                     SHOULDN'T IT BE: (state_spec, projection) tuple?
                                 dict (key=name, value=constraint_value or param dict)
                         if None, instantiate a single default State using constraint_value as state_spec
    - state_param_identifier (str): kw used to identify set of States in params;  must be one of:
        - INPUT_STATE
        - OUTPUT_STATE
    - constraint_value (2D np.array): set of 1D np.ndarrays used as default values and
        for compatibility testing in instantiation of State(s):
        - INPUT_STATE: self.instance_defaults.variable
        - OUTPUT_STATE: self.value
        ?? ** Note:
        * this is ignored if param turns out to be a dict (entry value used instead)
    - constraint_value_name (str):  passed to State._instantiate_state(), used in error messages
    - context (str)

    If state_list is None:
        - instantiate a default State using constraint_value,
        - place as the single entry of the list returned.
    Otherwise, if state_list is:
        - a single value:
            instantiate it (if necessary) and place as the single entry in an OrderedDict
        - a list:
            instantiate each item (if necessary) and place in a ContentAddressableList
    In each case, generate a ContentAddressableList with one or more entries, assigning:
        # the key for each entry the name of the OutputState if provided,
        #     otherwise, use MECHANISM<state_type>States-n (incrementing n for each additional entry)
        # the State value for each entry to the corresponding item of the Mechanism's state_type State's value
        # the dict to both self.<state_type>States and paramsCurrent[MECHANISM<state_type>States]
        # self.<state_type>State to self.<state_type>States[0] (the first entry of the dict)
    Notes:
        * if there is only one State, but the value of the Mechanism's state_type has more than one item:
            assign it to the sole State, which is assumed to have a multi-item value
        * if there is more than one State:
            the number of States must match length of Mechanisms state_type value or an exception is raised
    """

    state_entries = state_list
    # FIX: INSTANTIATE DICT SPEC BELOW FOR ENTRIES IN state_list

    # If no States were passed in, instantiate a default state_type using constraint_value
    if not state_entries:
        # assign constraint_value as single item in a list, to be used as state_spec below
        state_entries = constraint_value

        # issue warning if in VERBOSE mode:
        if owner.prefs.verbosePref:
            print("No {0} specified for {1}; default will be created using {2} of function ({3})"
                  " as its value".format(state_param_identifier,
                                         owner.__class__.__name__,
                                         constraint_value_name,
                                         constraint_value))

    # States should be either in a list, or possibly an np.array (from constraint_value assignment above):
    if isinstance(state_entries, (ContentAddressableList, list, np.ndarray)):

        # VALIDATE THAT NUMBER OF STATES IS COMPATIBLE WITH NUMBER OF CONSTRAINT VALUES
        num_states = len(state_entries)

        # Check that constraint_value is an indexable object, the items of which are the constraints for each State
        # Notes
        # * generally, this will be a list or an np.ndarray (either >= 2D np.array or with a dtype=object)
        # * for OutputStates, this should correspond to its value
        try:
            # Insure that constraint_value is an indexible item (list, >=2D np.darray, or otherwise)
            num_constraint_items = len(constraint_value)
        except:
            raise StateError("PROGRAM ERROR: constraint_value ({0}) for {1} of {2}"
                                 " must be an indexable object (e.g., list or np.ndarray)".
                                 format(constraint_value, constraint_value_name, state_type.__name__))

        # IMPLEMENTATION NOTE: NO LONGER VALID SINCE output_states CAN NOW BE ONE TO MANY OR MANY TO ONE
        #                      WITH RESPECT TO ITEMS OF constraint_value (I.E., owner.value)
        # If number of States exceeds number of items in constraint_value, raise exception
        if num_states > num_constraint_items:
            raise StateError("There are too many {}s specified ({}) in {} "
                                 "for the number of items ({}) in the {} of its function".
                                 format(state_param_identifier,
                                        num_states,
                                        owner.name,
                                        num_constraint_items,
                                        constraint_value_name))

        # If number of States is less than number of items in constraint_value, raise exception
        elif num_states < num_constraint_items:
            raise StateError("There are fewer {}s specified ({}) than the number of items ({}) "
                                 "in the {} of the function for {}".
                                 format(state_param_identifier,
                                        num_states,
                                        num_constraint_items,
                                        constraint_value_name,
                                        owner.name))

        # INSTANTIATE EACH STATE

        # Iterate through list or state_dict:
        # - instantiate each item or entry as state_type State
        # - get name, and use as key to assign as entry in self.<*>states
        states = ContentAddressableList(component_type=State_Base,
                                        name=owner.name+' ContentAddressableList of ' + state_param_identifier)

        # Instantiate State for entry in list or dict
        # Note: if state_entries is a list, state_spec is the item, and key is its index in the list
        for index, state_spec in state_entries if isinstance(state_entries, dict) else enumerate(state_entries):
            state_name = ""

            # State_entries is a dict, so use:
            # - entry index as State's name
            # - entry value as state_spec
            if isinstance(index, str):
                state_name = index
                state_constraint_value = constraint_value
                # Note: state_spec has already been assigned to entry value by enumeration above
                # MODIFIED 12/11/16 NEW:
                # If it is an "exposed" number, make it a 1d np.array
                if isinstance(state_spec, numbers.Number):
                    state_spec = np.atleast_1d(state_spec)
                # MODIFIED 12/11/16 END
                state_params = None

            # State_entries is a list
            else:
                if isinstance(state_spec, tuple):
                    if not len(state_spec) == 2:
                        raise StateError("List of {}s to instantiate for {} has tuple with more than 2 items:"
                                                  " {}".format(state_type.__name__, owner.name, state_spec))

                    state_spec, state_params = state_spec
                    if not (isinstance(state_params, dict) or state_params is None):
                        raise StateError("In list of {}s to instantiate for {}, second item of tuple "
                                                  "({}) must be a params dict or None:".
                                                  format(state_type.__name__, owner.name, state_params))
                else:
                    state_params = None

                # If state_spec is a string, then use:
                # - string as the name for a default State
                # - index (index in list) to get corresponding value from constraint_value as state_spec
                # - assign same item of constraint_value as the constraint
                if isinstance(state_spec, str):
                    # Use state_spec as state_name if it has not yet been used
                    if not state_name is state_spec and not state_name in states:
                        state_name = state_spec
                    # Add index suffix to name if it is already been used
                    # Note: avoid any chance of duplicate names (will cause current state to overwrite previous one)
                    else:
                        state_name = state_spec + '_' + str(index)
                    state_spec = constraint_value[index]
                    state_constraint_value = constraint_value[index]

                # FIX: 5/21/17  ADD, AND DEAL WITH state_spec AND state_constraint
                # elif isinstance(state_spec, dict):
                #     # If state_spec has NAME entry
                #     if NAME in state_spec:
                #         # If it has been used, add suffix to it
                #         if state_name is state_spec[NAME]:
                #             state_name = state_spec[NAME] + '_' + str(key)
                #         # Otherwise, use it
                #         else:
                #             state_name = state_spec[NAME]
                #     state_spec = ??
                #     state_constraint_value = ??


                # If state_spec is NOT a string, then:
                # - use default name (which is incremented for each instance in register_categories)
                # - use item as state_spec (i.e., assume it is a specification for a State)
                #   Note:  still need to get indexed element of constraint_value,
                #          since it was passed in as a 2D array (one for each State)
                else:
                    # # MODIFIED 9/3/17 OLD:
                    # # If only one State, don't add index suffix
                    # if num_states == 1:
                    #     state_name = 'Default_' + state_param_identifier[:-1]
                    # # Add incremented index suffix for each State name
                    # else:
                    #     state_name = 'Default_' + state_param_identifier[:-1] + "-" + str(index+1)
                    # MODIFIED 9/3/17 NEW:
                    # If only one State, don't add index suffix
                    if num_states == 1:
                        state_name = 'Default_' + state_param_identifier
                    # Add incremented index suffix for each State name
                    else:
                        state_name = 'Default_' + state_param_identifier + "-" + str(index+1)
                    # MODIFIED 9/3/17 END
                    # If it is an "exposed" number, make it a 1d np.array
                    if isinstance(state_spec, numbers.Number):
                        state_spec = np.atleast_1d(state_spec)

                    state_constraint_value = constraint_value[index]

            state = _instantiate_state(owner=owner,
                                       state_type=state_type,
                                       state_name=state_name,
                                       state_spec=state_spec,
                                       state_params=state_params,
                                       constraint_value=state_constraint_value,
                                       constraint_value_name=constraint_value_name,
                                       context=context)

            # Get name of state, and use as index to assign to states ContentAddressableList
            states[state.name] = state
        return states

    else:
        # This shouldn't happen, as MECHANISM<*>States was validated to be one of the above in _validate_params
        raise StateError("PROGRAM ERROR: {} for {} is not a recognized \'{}\' specification for {}; "
                         "it should have been converted to a list in Mechanism._validate_params)".
                         format(state_entries, owner.name, state_param_identifier, owner.__class__.__name__))


def _instantiate_state(owner,                  # Object to which state will belong
                      state_type,              # State subclass
                      state_name,              # Name for state (also used to refer to subclass in prompts)
                      state_spec,              # State subclass, object, spec dict, tuple, projection, value or str
                      state_params,            # params for state
                      constraint_value,        # Value used to check compatibility
                      constraint_value_name,   # Name of constraint_value's type (e.g. variable, output...)
                      context=None):
    """Instantiate a State of specified type, with a value that is compatible with constraint_value

    Constraint value must be a number or a list or tuple of numbers
    (since it is used as the variable for instantiating the requested state)

    If state_spec is a:
    + State class:
        implement default using constraint_value
    + State object:
        check owner is owner (if not, user is given options in _check_state_ownership)
        check compatibility of value with constraint_value
    + 2-item tuple: (only allowed for ParameterState spec)
        assign first item to state_spec
            if it is a string:
                test if it is a keyword and get its value by calling keyword method of owner's execute method
                otherwise, return None (suppress assignment of ParameterState)
        assign second item to STATE_PARAMS{PROJECTIONS:<projection>}
    + Projection object:
        assign constraint_value to value
        assign projection to STATE_PARAMS{PROJECTIONS:<projection>}
    + Projection class (or keyword string constant for one):
        assign constraint_value to value
        assign projection class spec to STATE_PARAMS{PROJECTIONS:<projection>}
    + specification dict for State (see XXX for context):
        check compatibility of STATE_VALUE with constraint_value
    + value:
        implement default using the value
    + str:
        test if it is a keyword and get its value by calling keyword method of owner's execute method
        # otherwise, return None (suppress assignment of ParameterState)
        otherwise, implement default using the string as its name
    Check compatibility with constraint_value
    If any of the conditions above fail:
        a default State of specified type is instantiated using constraint_value as value

    If state_params is specified, include as params arg with instantiation of State

    Returns a State or None
    """

    # IMPLEMENTATION NOTE: CONSIDER MOVING MUCH IF NOT ALL OF THIS TO State.__init__()

    # FIX: IF VARIABLE IS IN state_params EXTRACT IT AND ASSIGN IT TO constraint_value 5/9/17

    # VALIDATE ARGS
    if not inspect.isclass(state_type) or not issubclass(state_type, State):
        raise StateError("PROGRAM ERROR: state_type arg ({}) for _instantiate_state must be a State subclass".
                         format(state_type))
    if not isinstance(state_name, str):
        raise StateError("PROGRAM ERROR: state_name arg ({}) for _instantiate_state must be a string".
                             format(state_name))
    if not isinstance(constraint_value_name, str):
        raise StateError("PROGRAM ERROR: constraint_value_name arg ({}) for _instantiate_state must be a string".
                             format(constraint_value_name))

    state_params = state_params or {}

    # PARSE constraint_value
    constraint_dict = _parse_state_spec(owner=owner,
                                        state_type=state_type,
                                        state_spec=constraint_value,
                                        value=None,
                                        params=None)
    constraint_value = constraint_dict[VARIABLE]
    # constraint_value = constraint_dict[VALUE]

    # PARSE state_spec using constraint_value as default for value
    state_spec = _parse_state_spec(owner=owner,
                                   state_type=state_type,
                                   state_spec=state_spec,
                                   name=state_name,
                                   params=state_params,
                                   value=constraint_value)

    # state_spec is State object
    # - check that its value attribute matches the constraint_value
    # - check that its owner = owner
    # - if either fails, assign default State
    if isinstance(state_spec, state_type):
        # State initialization was deferred (owner or referenc_value was missing), so
        #    assign owner, variable, and/or reference_value if they were not specified
        if state_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:
            if not state_spec.init_args[OWNER]:
                state_spec.init_args[OWNER] = owner
                state_spec.init_args[VARIABLE] = owner.instance_defaults.variable[0]
            if not hasattr(state_spec, 'reference_value'):
                state_spec.reference_value = owner.instance_defaults.variable[0]
            state_spec._deferred_init()

        # Check that State's value is compatible with Mechanism's variable
        if iscompatible(state_spec.value, constraint_value):
            # Check that Mechanism is State's owner;  if it is not, user is given options
            state =  _check_state_ownership(owner, state_name, state_spec)
            if state:
                return state
            else:
                # State was rejected, and assignment of default selected
                state = constraint_value
        else:
            # State's value doesn't match constraint_value, so assign default
            if owner.verbosePref:
                warnings.warn("Value of {} for {} ({}, {}) does not match expected ({}); "
                              "default {} will be assigned)".
                              format(state_type.__name__,
                                     owner.name,
                                     state_spec.name,
                                     state_spec.value,
                                     constraint_value,
                                     state_type.__name__))
            # state = constraint_value
            state_variable = constraint_value

    # Otherwise, state_spec should now be a state specification dict
    state_variable = state_spec[VARIABLE]
    state_value = state_spec[VALUE]

    # Check that it's variable is compatible with constraint_value, and if not, assign the latter as default variable
    if constraint_value is not None and not iscompatible(state_variable, constraint_value):
        if owner.prefs.verbosePref:
            print("{} is not compatible with constraint value ({}) specified for {} of {};  latter will be used".
                  format(VARIABLE, constraint_value, state_type, owner.name))
        state_variable = constraint_value
    # else:
    #     constraint_value = state_variable

    # INSTANTIATE STATE:
    # Note: this will be either a default State instantiated using constraint_value as its value
    #       or one determined by a specification dict, depending on which of the following obtained above:
    # - state_spec was a 2-item tuple
    # - state_spec was a specification dict
    # - state_spec was a value
    # - value of specified State was incompatible with constraint_value
    # - owner of State was not owner and user chose to implement default
    # IMPLEMENTATION NOTE:
    # - setting prefs=NotImplemented causes TypeDefaultPreferences to be assigned (from ComponentPreferenceSet)
    # - alternative would be prefs=owner.prefs, causing state to inherit the prefs of its owner;

    #  Convert constraint_value to np.array to match state_variable (which, as output of function, will be an np.array)
    constraint_value = convert_to_np_array(constraint_value,1)

    # Implement default State
    state = state_type(owner=owner,
                       reference_value=constraint_value,
                       variable=state_variable,
                       name=state_spec[NAME],
                       params=state_spec[PARAMS],
                       prefs=None,
                       context=context)

# FIX LOG: ADD NAME TO LIST OF MECHANISM'S VALUE ATTRIBUTES FOR USE BY LOGGING ENTRIES
    # This is done here to register name with Mechanism's stateValues[] list
    # It must be consistent with value setter method in State
# FIX LOG: MOVE THIS TO MECHANISM STATE __init__ (WHERE IT CAN BE KEPT CONSISTENT WITH setter METHOD??
#      OR MAYBE JUST REGISTER THE NAME, WITHOUT SETTING THE
# FIX: 2/17/17:  COMMENTED THIS OUT SINCE IT CREATES AN ATTRIBUTE ON OWNER THAT IS NAMED <state.name.value>
#                NOT SURE WHAT THE PURPOSE IS
#     setattr(owner, state.name+'.value', state.value)

    return state

def _check_parameter_state_value(owner, param_name, value):
    """Check that parameter value (<ParameterState>.value) is compatible with value in paramClassDefault

    :param param_name: (str)
    :param value: (value)
    :return: (value)
    """
    default_value = owner.paramClassDefaults[param_name]
    if iscompatible(value, default_value):
        return value
    else:
        if owner.prefs.verbosePref:
            print("Format is incorrect for value ({0}) of {1} in {2};  default ({3}) will be used.".
                  format(value, param_name, owner.name, default_value))
        return default_value

def _check_state_ownership(owner, param_name, mechanism_state):
    """Check whether State's owner is owner and if not offer options how to handle it

    If State's owner is not owner, options offered to:
    - reassign it to owner
    - make a copy and assign to owner
    - return None => caller should assign default

    :param param_name: (str)
    :param mechanism_state: (State)
    :param context: (str)
    :return: (State or None)
    """

    if mechanism_state.owner != owner:
        reassign = input("\nState for \'{0}\' parameter, assigned to {1} in {2}, already belongs to {3}. "
                         "You can reassign it (r), copy it (c), or assign default (d):".
                         format(mechanism_state.name, param_name, owner.name,
                                mechanism_state.owner.name))
        while reassign != 'r' and reassign != 'c' and reassign != 'd':
            reassign = input("\nReassign (r), copy (c), or default (d):".
                             format(mechanism_state.name, param_name, owner.name,
                                    mechanism_state.owner.name))

            if reassign == 'r':
                while reassign != 'y' and reassign != 'n':
                    reassign = input("\nYou are certain you want to reassign it {0}? (y/n):".
                                     format(param_name))
                if reassign == 'y':
                    # Note: assumed that parameters have already been checked for compatibility with assignment
                    return mechanism_state

        # Make copy of state
        if reassign == 'c':
            import copy
            # # MODIFIED 10/28/16 OLD:
            # mechanism_state = copy.deepcopy(mechanism_state)
            # MODIFIED 10/28/16 NEW:
            # if owner.verbosePref:
                # warnings.warn("WARNING: at present, 'deepcopy' can be used to copy States, "
                #               "so some components of {} might be missing".format(mechanism_state.name))
            print("WARNING: at present, 'deepcopy' can be used to copy states, "
                  "so some Components of {} assigned to {} might be missing".
                  format(mechanism_state.name, append_type_to_name(owner)))
            mechanism_state = copy.copy(mechanism_state)
            # MODIFIED 10/28/16 END

        # Assign owner to chosen state
        mechanism_state.owner = owner
    return mechanism_state


def _check_projection_sender_compatability(owner, projection_type, sender_type):
    from PsyNeuLink.Components.States.OutputState import OutputState
    from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
    from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
    from PsyNeuLink.Components.States.ModulatorySignals.GatingSignal import GatingSignal
    from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
    from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
    from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection
    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base

    if issubclass(projection_type, MappingProjection) and issubclass(sender_type, (OutputState,
                                                                                   ProcessingMechanism_Base)):
        return
    if issubclass(projection_type, LearningProjection) and issubclass(sender_type, LearningSignal):
        return
    if issubclass(projection_type, ControlProjection) and issubclass(sender_type, ControlSignal):
        return
    if issubclass(projection_type, GatingProjection) and issubclass(sender_type, GatingSignal):
        return
    else:
        raise StateError("Illegal specification of a {} for {} of {}".
                         format(sender_type, owner.__class__.__name__, owner.owner.name))


def _parse_state_type(owner, state_spec):
    """Determine State type for state_spec and return type

    Determine type from context and/or type of state_spec if the latter is not a `State <State>` or `Mechanism
    <Mechanism>`.
    """

    # State class reference
    if isinstance(state_spec, State):
        return type(state_spec)

    # keyword for a State or name of a standard_output_state
    if isinstance(state_spec, str):

        # State keyword
        if state_spec in state_type_keywords:
            import sys
            return getattr(sys.modules['PsyNeuLink.Components.States.'+state_spec], state_spec)

        # standard_output_state
        if hasattr(owner, STANDARD_OUTPUT_STATES):
            # check if string matches the name entry of a dict in standard_output_states
            # item = next((item for item in owner.standard_output_states.names if state_spec is item), None)
            # if item is not None:
            #     # assign dict to owner's output_state list
            #     return owner.standard_output_states.get_dict(state_spec)
            # from PsyNeuLink.Components.States.OutputState import StandardOutputStates
            if owner.standard_output_states.get_state_dict(state_spec):
                from PsyNeuLink.Components.States.OutputState import OutputState
                return OutputState

    # State specification dict
    if isinstance(state_spec, dict):
        if STATE_TYPE in state_spec:
            if not inspect.isclass(STATE_TYPE) and issubclass(state_spec[STATE_TYPE], State):
                raise StateError("STATE entry of state specification for {} ({})"
                                 "is not a State or type of State".
                                 format(owner.name, state_spec[STATE]))
            return state_spec[STATE]

    # # Mechanism specification (State inferred from context)
    # if isinstance(state_spec, Mechanism):

    # # Projection specification (State inferred from context)
    # if isinstance(state_spec, Projection):

    # # 2-item specification (State inferred from context)
    # if isinstance(state_spec, tuple):
    #     _is_legal_state_spec_tuple(owner, state_spec)
    #     tuple_spec = state_spec[1]
    #     if isinstance(tuple_spec, State):
    #         # InputState specified as 2nd item of tuple must be a destination, so choose State based on owner:
    #         if isinstance(owner, ProcessingMechanism)
    #             if isinstance(tuple_spec, InputState):
    #                 return OutputState
    #             if isinstance(tuple_spec, OutputState):
    #                 return InputState
    #     else:
    #         # Call recursively to handle other types of specs
    #         return _parse_state_type(owner, tuple_spec)

    raise StateError("{} is not a legal State specification for {}".format(state_spec, owner.name))


# FIX 5/23/17:  UPDATE TO ACCOMODATE (param, ControlSignal) TUPLE
@tc.typecheck
def _parse_state_spec(owner,
                      state_type:_is_state_type,
                      state_spec,
                      name:tc.optional(str)=None,
                      variable=None,
                      value=None,
                      # projections:tc.any(list, bool)=[],
                      # modulatory_projections:tc.any(list,bool)=[],
                      params=None,
                      force_dict=False):

    """Return either State object or State specification dict for state_spec

    If state_spec is or resolves to a State object:
        if force_dict is False:  return State object
        if force_dict is True: parse into State specification_dictionary
            (replacing any components with their id to avoid problems with deepcopy)
    Otherwise, return State specification dictionary using arguments provided as defaults
    Warn if variable is assigned the default value, and verbosePref is set on owner.
    **value** arg should generally be a constraint for the value of the State;
        if state_spec is a Projection, and method is being called from:
            InputState, value should be the projection's value;
            ParameterState, value should be the projection's value;
            OutputState, value should be the projection's variable
    Any entries with keys other than XXX are moved to entries of the dict in the PARAMS entry
    """

    # # IMPLEMENTATION NOTE:  ONLY CALLED IF force_dict=True;  CAN AVOID BY NEVER SETTING THAT OPTION TO True
    # #                       STILL NEEDS WORK: SEEMS TO SET PARAMS AND SO CAUSES CALL TO assign_params TO BAD EFFECT
    # # Get convert state object into State specification dictionary,
    # #     replacing any set, dict or Component with its id to avoid problems with deepcopy
    # @tc.typecheck
    # def _state_dict(state:State):
    #     @tc.typecheck
    #     # Checks if item is Component and returns its id if it is
    #     def filter_params(item):
    #         if isinstance(item, (set, dict, Component)):
    #             item = id(item)
    #         return item
    #     if hasattr(state, 'params') and state.params:
    #         for param in state.params:
    #             if isinstance(param, collections.Iterable) and not isinstance(param, str):
    #                 for index, item in param if isinstance(param, dict) else enumerate(param):
    #                     state.params[param][index] = filter_params(item)
    #             else:
    #                 state.params[param] = filter_params(state.params[param])
    #     return dict(**{NAME:state.name,
    #                   VARIABLE:variable,
    #                   VALUE:state.value,
    #                   # PARAMS:{PROJECTIONS:state.pathway_projections}})
    #                   PARAMS:state.params})

    # Validate that state_type is a State class
    if not inspect.isclass(state_type) or not issubclass(state_type, State):
        raise StateError("\'state_type\' arg ({}) must be a sublcass of {}".format(state_type,
                                                                                   State.__name__))
    state_type_name = state_type.__name__

    # State object:
    # - check that it is of the specified type and, if so:
    #     - if force_dict is False, return the primary state object
    #     - if force_dict is True, get state's attributes and return their values in a State specification dictionary

    if isinstance(state_spec, State):
        if isinstance(state_spec, state_type):
            # if force_dict:
            #     return _state_dict(state_spec)
            # else:
            #     return state_spec
            return state_spec
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type_name))

    # Mechanism object:
    # - call owner to get primary state of specified type;
    # - if force_dict is False, return the primary state object
    # - if force_dict is True, get primary state's attributes and return their values in a state specification dict
    if isinstance(state_spec, Mechanism):
        primary_state = state_spec._get_primary_state(state_type)
        # if force_dict:
        #     return _state_dict(primary_state)
        # else:
        #     return primary_state
        return primary_state

    # # Avoid modifying any objects passed in via state_spec
    # state_spec = copy.deepcopy(state_spec)

    # params = params or {}

    if params:
        # If variable is specified in state_params, use that
        if VARIABLE in params and params[VARIABLE] is not None:
            variable = self._update_variable(params[VARIABLE])

    # Create default dict for return
    state_dict = {NAME: name,
                  VARIABLE: variable,
                  VALUE: value,
                  PARAMS: params,
                  STATE_TYPE: state_type
                  }

    # State class
    if inspect.isclass(state_spec) and issubclass(state_spec, State):
        if state_spec is state_type:
            state_dict[VARIABLE] = state_spec.ClassDefaults.variable
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type_name))

    # # State object [PARSED INTO DICT HERE]
    # elif isinstance(state_spec, State):
    #     if state_spec is state_type:
    #         name = state_spec.name
    #         # variable = state_spec.value
    #         # variable = state_spec.ClassDefaults.variable
    #         value = state_spec.value
    #         modulatory_projections =  state_spec.mod_projections
    #         params = state_spec.user_params.copy()
    #     else:
    #         raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
    #                          "class of state being instantiated ({})".format(state_spec, state_type))

    # Specification dict
    # - move any entries other than for standard args into dict in params entry
    elif isinstance(state_spec, dict):
        # Dict has a single entry in which the key is not a recognized keyword,
        #    so assume it is of the form {<STATE_NAME>:<STATE SPECIFICATION DICT>}:
        #    assign STATE_NAME as name, and return parsed SPECIFICATION_DICT
        if len(state_spec) == 1 and list(state_spec.keys())[0] not in (state_keywords | STANDARD_ARGS):
            name, state_spec = list(state_spec.items())[0]
            state_dict = _parse_state_spec(owner=owner,
                                           state_type=state_type,
                                           state_spec=state_spec,
                                           name=name,
                                           variable=variable,
                                           value=value,
                                           # projections=projections,
                                           # modulatory_projections=modulatory_projections,
                                           # params=params)
                                           )
            if state_dict[PARAMS]:
                params = params or {}
                # params.update(state_dict[PARAMS])
                # Use name specified as key in state_spec (overrides one in SPEFICATION_DICT if specified):
                state_dict[PARAMS].update(params)

        else:
            # Error if STATE_TYPE is specified in dict and not the same as the state_spec specified in call to method
            if STATE_TYPE in state_spec and state_spec[STATE_TYPE] != state_type:
                raise StateError("PROGRAM ERROR: STATE_TYPE entry in State specification dictionary for {} ({}) "
                                 "is not the same as one specified in call to _parse_state_spec ({})".
                                 format(owner.name, state_spec[STATE_TYPE, state_type]))
            # Warn if VARIABLE was not in dict
            if not VARIABLE in state_spec and owner.prefs.verbosePref:
                print("{} missing from specification dict for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, state_spec))
            # Move all params-relevant entries from state_spec into params
            for spec in [param_spec for param_spec in state_spec.copy()
                         if not param_spec in STANDARD_ARGS]:
                params = params or {}
                params[spec] = state_spec[spec]
                # MODIFIED 6/5/17 OLD: [REINSTATED, BUT CAUSING TROUBLE IN STROOP TEST SCRIPT]
                del state_spec[spec]
                # MODIFIED 6/5/17 END
            state_dict.update(state_spec)
            # state_dict = state_spec
            if params:
                if state_dict[PARAMS] is None:
                    state_dict[PARAMS] = {}
                state_dict[PARAMS].update(params)

    # # 2-item tuple (spec, projection)
    # 2-item tuple (spec, Component)
    elif isinstance(state_spec, tuple):
        _is_legal_state_spec_tuple(owner, state_spec, state_type_name)
        # Put projection spec from second item of tuple in params
        params = params or {}
        # FIX 5/23/17: NEED TO HANDLE NON-MODULATORY PROJECTION SPECS
        params.update({PROJECTIONS:[state_spec[1]]})

        # Parse state_spec in first item of tuple (without params)
        state_dict = _parse_state_spec(owner=owner,
                                       state_type=state_type,
                                       state_spec=state_spec[0],
                                       name=name,
                                       variable=variable,
                                       value=value,
                                       # projections=projections,
                                       params={})
        # Add params (with projection spec) to any params specified in first item of tuple
        if state_dict[PARAMS] is None:
            state_dict[PARAMS] = {}
        state_dict[PARAMS].update(params)

    # Projection class, object, or keyword:
    #     set variable to value and assign projection spec to PROJECTIONS entry in params
    # IMPLEMENTATION NOTE:  It is the caller's responsibility to assign the value arg
    #                           appropriately for the state being requested, for:
    #                               InputState, projection's value;
    #                               ParameterState, projection's (= parameter's) value;
    #                               OutputState, projection's variable .
    # Don't allow matrix keywords -- force them to be converted from a string into a value (below)
    elif _is_projection_spec(state_spec, include_matrix_keywords=False):
        # state_spec = state_variable
        state_dict[VARIABLE] =  value
        if state_dict[PARAMS] is None:
            state_dict[PARAMS] = {}
        state_dict[PARAMS].update({PROJECTIONS:[state_spec]})

    # string (keyword or name specification)
    elif isinstance(state_spec, str):
        # Check if it is a keyword
        spec = get_param_value_for_keyword(owner, state_spec)
        # A value was returned, so use value of keyword as variable
        if spec is not None:
            state_dict[VARIABLE] = spec
            # NOTE: (7/26/17 CW) This warning below may not be appropriate, since this routine is run if the
            # matrix parameter is specified as a keyword, which may be intentional.
            if owner.prefs.verbosePref:
                print("{} not specified for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, value))
        # It is not a keyword, so treat string as the name for the state
        else:
            state_dict[NAME] = state_spec

    # function; try to resolve to a value, otherwise return None to suppress instantiation of state
    elif isinstance(state_spec, function_type):
        state_dict[VALUE] = get_param_value_for_function(owner, state_spec)
        if state_dict[VALUE] is None:
            # return None
            raise StateError("PROGRAM ERROR: state_spec for {} of {} is a function ({}), "
                             "but it failed to return a value".format(state_type_name, owner.name, state_spec))

    # value, so use as variable of input_state
    elif is_value_spec(state_spec):
        state_dict[VARIABLE] = state_spec
        state_dict[VALUE] = state_spec

    elif state_spec is None:
        # pass
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is None".format(state_type_name, owner.name))

    else:
        if name and hasattr(owner, name):
            owner_name = owner.name
        else:
            owner_name = owner.__class__.__name__
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is an unrecognized specification ({})".
                         format(state_type_name, owner.name, state_spec))


    # If variable is none, use value:
    if state_dict[VARIABLE] is None:
        state_dict[VARIABLE] = state_dict[VALUE]

    # # Add STATE_TYPE entry to state_dict
    # state_dict[STATE_TYPE] = state_type

    return state_dict


def _is_legal_state_spec_tuple(owner, state_spec, state_type_name=None):

    state_type_name = state_type_name or STATE

    if len(state_spec) != 2:
        raise StateError("Tuple provided as state_spec for {} of {} ({}) must have exactly two items".
                         format(state_type_name, owner.name, state_spec))
    # IMPLEMENTATION NOTE: Mechanism allowed in tuple to accomodate specification of param for ControlSignal
    if not (_is_projection_spec(state_spec[1]) or isinstance(state_spec[1], (Mechanism, State))):
        raise StateError("2nd item of tuple in state_spec for {} of {} ({}) must be a specification "
                         "for a Mechanism, State, or Projection".
                         format(state_type_name, owner.__class__.__name__, state_spec[1]))

