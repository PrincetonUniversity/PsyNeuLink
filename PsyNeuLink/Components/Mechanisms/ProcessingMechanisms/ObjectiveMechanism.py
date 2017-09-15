# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ObjectiveMechanism ****************************************************

"""

Overview
--------

An ObjectiveMechanism is a `ProcessingMechanism <ProcessingMechanism>` that monitors the `OutputStates <OutputState>`
of one or more other ProcessingMechanisms specified in its `monitor <ObjectiveMechanism.monitor>` attribute,
and evaluates them using its `function <ObjectiveMechanism.function>`. The result of the evaluation is placed in the
ObjectiveMechanism's `primary OutputState <OutputState_Primary>`.  ObjectiveMechanisms are typically used closely
with (and often created automatically by) `AdaptiveMechanisms <AdaptiveMechanism>`.

.. _ObjectiveMechanism_Creation:

Creating an ObjectiveMechanism
------------------------------

ObjectiveMechanisms are often created automatically when other PsyNeuLink components are created (in particular,
AdaptiveMechanisms such as `LearningMechanisms <LearningMechanism_Creation>` and
`ControlMechanisms <ControlMechanism_Creation>`.  An ObjectiveMechanism can also be created directly by calling its
constructor.  The two primary attributes used to define an ObjectiveMechanism are its `monitored_output_states
<ObjectiveMechanism.monitored_output_states>` and `input_states <ObjectiveMechanism.input_states>` attributes, specified
using the corresponding arguments of its construtor as described below.

.. _ObjectiveMechanism_Monitored_output_states:

Monitored Values
~~~~~~~~~~~~~~~~

The **monitored_output_states** argument of the constructor specifies the OutputStates it monitors.  The assignment of
the OutputStates to be monitored can also be passed from the **objective_mechanism** argument of the constructor for a
`ControlMechanism`, or the **monitor_for_control** argument of a `System` for which the ControlMechanism is a
`controller <System_Base.controller>`.  The **monitored_output_states** argument can be specified as any one of a list
of the following:

COMMENT:
Note that some forms of specification may
depend on specifications made for the OutputState referenced, the Mechanism to which it belongs, and/or the Process
or System to which that Mechanism
belongs. These interactions (and the precedence afforded to each) are described below.
COMMENT

  * **OutputState** -- a reference to the `OutputState <OutputState>` of a Mechanism.  This creates a
    `MappingProjection` from it to the corresponding InputState in `input_states <ObjectiveMechanism.input_states>`.
  COMMENT:
      TBI
      Note that an outputState can be *excluded* from being monitored by assigning `None` as the value of its
      `monitoring_status` attribute.  This specification takes precedence over any others;  that is, it suppresses
      monitoring of that OutputState, irrespective of any other specifications that might otherwise apply to that
      OutputState, including those described below.
  ..
  TBI
  * **Mechanism** -- by default, the Mechanism's `primary OutputState <OutputState_Primary>` is used.  However,
    if the Mechanism has any OutputStates specified in its `monitored_states` attribute, those are used (except for
    any that specify `None` as their `monitoring_status`). This specification takes precedence over any of the other
    types listed below:  if it is `None`, then none of that Mechanism's OutputStates are monitored; if it
    specifies OutputStates to be monitored, those are monitored even if they do not satisfy any of the conditions
    described in the specifications below.
  COMMENT
  ..
  * **Mechanism** -- the Mechanism's `primary OutputState <OutputState_Primary>` is used.

  * **string** -- this can be used as a "placemarker", to specify that an `InputState` be created for an OutputState
    to be monitored that will be identified later.  The string is used as the name of the InputState when it is created.

  .. _ObjectiveMechanism_OutputState_Tuple:

  * **monitored_output_states tuple** -- this is a convenience notation that can be used to compactly specify a weight
    and exponent for an OutputState' (see `example <ObjectiveMechanism_OutputState_Tuple_Example>`). Each tuple must
    have the three following items in the order listed:

        * any of the specifications above -- if it is a string, the weight and exponent specified in the tuple
          (see below) will be assigned along with the string as the name of the "placemarker" InputState when it is
          created.
        |
        * a weight -- must be an integer or a float; multiplies the value of the OutputState before being combined with
          others by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.
        |
        * an exponent -- must be an integer or float; exponentiates the value of the OutputState before being combined
          with others by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.

    .. _ObjectiveMechanism_Weights_And_Exponents:

    .. note::
       The weight and exponent specifications in a tuple are used to specify the `weight <InputState.weight>` and
       `exponent <InputState.exponent>` of the corresponding InputState of the ObjectiveMechanism;  therefore,
       any values specified there take precedence over any other weight and/or exponent specifications of the
       InputStates for the ObjectiveMechanism.

  .. _ObjectiveMechanism_OutputState_Specification_Dictionary:

  * **monitored_value specification dictionary** -- this the most flexible form of specification, that can be used
    to specify one or more OutputStates of a Mechanism by their name

        * *MECHANISM*:`Mechanism <Mechanism>`
            this enty must be included in the dictionary;

        * *OUTPUT_STATES*:List[<str or any of the specifications above>, ...]
            here, a string can be used anywhere an OutputState would have been specified as the name
            of an OutputState belonging to the *MECHANISM* entry.

        If the dictionary includes only the *MECHANISM* entry, then (as with a standalone Mechanism specification),
        the Mechanism's `primary OutputState <OutputState_Primary>` is used.  If a string is used for any items
        of the list in the *OUTPUT_STATES* entry, and it is not the name of the Mechanism in the *MECHANISM* entry,
        it will be treated as an "unbound" string specification.

If an OutputState to be monitored is specified at the time the ObjectiveMechanism is created, or the specification
can be resolved to an OutputState, a `MappingProjection` is automatically created from it to the corresponding
InputState of the ObjectiveMechanism using `AUTO_ASSIGN_MATRIX` as its `matrix <MappingProjection.matrix>` parameter.
If the OutputState can't be determined, no MappingProjection is assigned, and this must be done by some other means;
any values assigned to the ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
attribute that are not associated with an OutputState at the time the ObjectiveMechanism is executed are ignored.


.. _ObjectiveMechanism_InputStates::

InputStates
~~~~~~~~~~~

When an ObjectiveMechanism is created, an InputState is created for each of the OutputStates specified in its
**monitored_output_states** argument, and a `MappingProjection` is assigned from each of those to the corresponding
InputState.  By default, the `value <InputState.value>` of each InputState uses the format of the `value
<OutputState.value>`  of its corresponding OutputState, and the MappingProjection between them uses an
`IDENTITY_MATRIX`.  However, the **input_states** argument can be used to specify the names of the InputStates
and/or customize their lengths (e.g., specify one that is different from the OutputState that projects to it).  Any of
the forms used for `specifying InputStates <InputState_Specification>` can be used in  the **input_states** argument,
including a list of strings (to be used as names for the InputStates), values (used to specify their length),
or `State Specification Dictionaries <State_Specification>`, with the following constraints:

  * the number of InputStates specified in the **input_states** argument must equal the number of OutputStates specified
    in the **monitored_output_states** argument (see the `examples
    <ObjectiveMechanism_Monitored_output_states_Examples>` below);
  ..
  * the `variable <InputState.variable>` of each must also be of the same type as the `value <OutputState.value>` of
    the corresponding OutputState, however their lengths can differ;  in that case, by default,
    a `FULL_CONNECTIVITY_MATRIX` is used to create the MappingProjection between them (although this too can be
    customized using the *PROJECTION* entry of a `State specification dictionary <State_Specification>` for the
    InputState in the **input_states** argument;
  ..
  * a `weight and/or exponent <InputState_Weights_And_Exponents>` can be specified for each InputState (e.g., for use
    by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`;  however, these are superseded by any weights
    and/or exponents specified in the **monitored_output_states** argument of the ObjectiveMechanism's constructor
    (see `ObjectiveMechanism_Monitored_output_states` and `note <ObjectiveMechanism_Weights_And_Exponents>` above).

COMMENT:
  * **string**, **value** or **dict** -- these can be used as placemarkers for a state to be monitored, that will be
    instantiated later (for example, for the TARGET input of a Composition).  If a string is specified, it is used as
    the default name of the corresponding InputState (specified in the `input_states <ObjectiveMechanism.input_states>`
    attribute of the ObjectiveMechanism). If a value is specified, it is used as the default value for the corresponding
    InputState.  If a dict is specified, it must have a single entry, the key of which is used as a string
    specification -- i.e., as the name of the InputState -- and the value as its value specification.
COMMENT

.. _ObjectiveMechanism_Structure:

Structure
---------

.. _ObjectiveMechanism_Input:

Input
~~~~~

=An ObjectiveMechanism has one `InputState <InputState>` for each of the OutputStates specified in its
**monitored_output_states** argument (see `ObjectiveMechanism_Monitored_output_states`). Each InputState receives a
`MappingProjection` from the corresponding OutputState, the values of which are used by the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` to generate the value of its *OUTCOME* `OutputState
<ObjectiveMechanism_Output>`.  The InputStates are listed in the ObjectiveMechanism's `input_states
<ObjectiveMechanism.input_states>` attribute, and the OutputStates from which they receive projections are listed in
the same order its `monitored_output_states  <ObjectiveMechanism.monitored_output_states>` attribute.  If any `weights
and/or exponents are specified <ObjectiveMechanism_Monitored_output_states>` for either the ObjectiveMechanism's
`monitored_output_states <ObjectiveMechanism.monitored_output_states>` or its `input_states`, the former take
precedence over the latter.  These are assigned to the weights and/or exponents attributes of the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` if the function implements these attributes;  if so, the function applies the
weights and/or exponents specified to the corresponding InputState `value <InputState.value>`\\s before it combining
these to generate the ObjectiveMechanism's `output <ObjectiveMechanism_Output>`.

.. _ObjectiveMechanism_Function:

Function
~~~~~~~~

The ObjectiveMechanism's `function <ObjectiveMechanism.function>` uses the values of its InputStates to compute an
`objective (or "loss") function <https://en.wikipedia.org/wiki/Loss_function>`_, that is assigned as the value of its
*OUTCOME* `OutputState <ObjectiveMechanism_Output>`.  By default, it uses a `LinearCombination` function to
sum the values of the values of the OutputStates listed in `monitored_output_states`. However, this can be configured to
calculate differences, ratios, etc. (see `example <ObjectiveMechanism_Weights_and_Exponents_Example>` below).  It can
also be replaced with any `CombinationFunction`, or any python function that takes a nd array as its input (with a
number of items in axis 0 equal to the number of the ObjectiveMechanism's InputStates), and generates a 1d array as
its result. If it implements :keyword:`weight` and/or :keyword:`exponent` attributes,
those are assigned from `weight <InputState.weight>` and `exponent <InputState.exponent>` attributes of its
`input_states <ObjectiveMechanism.input_states>` (listed in its `monitored_output_states_weights_and_exponents
<ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute);  otherwise, they are ignored.

.. _ObjectiveMechanism_Output:

Output
~~~~~~

The `primary OutputState <OutputState_Primary>` of an Objective mechanism is a 1d array, named *OUTCOME*, that is the
result of its `function <ObjectiveMechanism.function>` (as described above).


.. _ObjectiveMechanism_Execution:

Execution
---------

When an ObjectiveMechanism is executed, it updates its input_states with the values of the OutputStates listed in
its `monitored_output_states` attribute, and then uses its `function <ObjectiveMechanism.function>` to
evaluate these.  The result is assigned as to its `value <ObjectiveMechanism.value>` attribute as the value of its
`primary OutputState <OutputState_Primary>`.


.. _ObjectiveMechanism_Examples:

Examples
--------

.. _ObjectiveMechanism_Monitored_output_states_Examples:

*Specifying* **monitored_output_states**

The use of default_variable to override a specification in `monitored_output_states` can be useful in some situations.
For example, for `Reinforcement Learning <Reinforcement>`, an ObjectiveMechanism is used to monitor an action
selection Mechanism.  In the example below, the latter uses a `TransferMechanism` with the `SoftMax` function (and the
`PROB <Softmax.PROB>` as its output format) to select the action.  This generates a vector with a single non-zero
value, which designates the predicted reward for the selected action.  Because the output is a vector,
by default the InputState of the ObjectiveMechanism created to monitor it will also be a vector.  However, the
ObjectiveMechanism requires that this be a single value, that it can compare with the value of the reward Mechanism.
This can be dealt with by using `default_variable` in the constructor of the ObjectiveMechanism, to force
the InputState for the ObjectiveMechanism to have a single value, as in the example below::

    my_action_select_mech = TransferMechanism(default_variable = [0,0,0], function=SoftMax(output=PROB))

    my_reward_mech = TransferMechanism(default_variable = [0])

    my_objective_mech = ObjectiveMechanism(monitored_output_states = [my_action_select_mech, my_reward_mech])

Note that the OutputState for the ``my_action_selection`` and ``my_reward_mech`` are specified
in `monitored_output_states`.  If that were the only specification, the InputState created for ``my_action_select_mech``
would be a vector of length 3.  This is overridden by specifying `default_variable` as an array with two
single-value arrays (one corresponding to ``my_action_select_mech`` and the other to ``my_reward_mech``).  This forces
the InputState for ``my_action_select_mech`` to have only a single element which, in turn, will cause a
MappingProjection to be created from  ``my_action_select_mech`` to the ObjectiveMechanism's InputState using a
`FULL_CONNECTIVITY_MATRIX` (the one used for `AUTO_ASSIGN_MATRIX` when the sender and receiver have values of
different lengths).  This produces the desired effect, since the action selected is the only non-zero value in the
output of ``my_action_select_mech``, and so the `FULL_CONNECTIVITY_MATRIX` will combine it with zeros (the other values
in the vector), and so its value will be assigned as the value of the corresponding InputState in the
ObjectiveMechanism.  Another option would have been to customize the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` to convert the output of ``my_action_select_mech`` to a length 1 vector, though
this would have been more involved.  The next example describes a simple case of customizing the ObjectiveMechanism's
`function <ObjectiveMechanism.function>`, however more sophisticated ones are possible, just as the one just suggested.

.. _ObjectiveMechanism_Weights_and_Exponents_Example:

*Customizing the ObjectiveMechanism's function*

The simplest way to customize the `function <ObjectiveMechanism.function>` of an ObjectiveMechanism is to
parametrize its default function (`LinearCombination`).  In the example below, the ObjectiveMechanism used in the
previous example is further customized to subtract the value of the action selected from the value of the reward::

    my_objective_mech = ObjectiveMechanism(default_variable = [[0],[0]],
                                           monitored_output_states = [my_action_select_mech, my_reward_mech],
                                           function=LinearCombination(weights=[[-1], [1]]))

This is done by specifying the `weights <LinearCombination.weights>` parameter of the `LinearCombination` function,
with two values [-1] and [1] corresponding to the two items in `monitored_output_states` (and `default_variable`).
This will multiply the value from ``my_action_select_mech`` by -1 before adding it to (and thus subtracting it from)
the value of ``my_reward_mech``.  Notice that the weight for ``my_reward_mech`` had to be specified, even though it
is using the default value (1);  whenever a weight and/or exponent parameter is specified, there must be an entry for
every item of the function's variable.  The `operation <LinearCombination.operation>` and `exponents
<LinearCombination.exponents>` parameters of `LinearCombination` can be used similarly, and together, to multiply and
divide quantities.

.. _ObjectiveMechanism_OutputState_Tuple_Example:

As a conveninence notation, weights and exponents can be included with the specification of the OutputState itself, in
the **monitored_output_states** argument, by placing them in a tuple with the OutputState (see `monitored_output_states
tuple <ObjectiveMechanism_OutputState_Tuple>`).  The following example specifies the example same ObjectiveMechanism
as the previous example::

    my_objective_mech = ObjectiveMechanism(default_variable = [[0],[0]],
                                           monitored_output_states = [(my_action_select_mech, -1, 1), my_reward_mech])

This specifies that ``my_action_select_mech`` should be assigned a weight of -1 and an exponent of 1 when it is
submitted to the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  Notice that the exponent had to be
included, even though it is the default value;  when a tuple is used, the weight and exponent values must both be
specified.  Notice also that ``my_reward_mech`` does not use a tuple, so it will be assigned defaults for both the
weight and exponent parameters.


The following example uses a dictionary to specify the **monitored_output_states** argument, allowing several
OutputStates for the same Mechanism to be specified more easily, each by name rather than by full reference
(which is required if they are specified on their own or in a tuple::

    my_objective_mech = ObjectiveMechanism(monitored_output_states=[Reward,
                                                            {MECHANISM: Decision,
                                                             OUTPUT_STATES: [PROBABILITY_UPPER_THRESHOLD,
                                                                             (RESPONSE_TIME, 1, -1)]}])

Note that, as shown in this example, the tuple format can still be used for each individual OutputState in the list
assigned to the *OUTPUT_STATES* entry.



.. _ObjectiveMechanism_Class_Reference:

Class Reference
---------------

"""
import warnings

import typecheck as tc
import numbers


from PsyNeuLink.Components.Component import InitStatus
from PsyNeuLink.Components.Functions.Function import LinearCombination
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base, MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.ShellClasses import Mechanism, State
from PsyNeuLink.Components.States.State import _parse_state_spec
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import OutputState, PRIMARY_OUTPUT_STATE, standard_output_states
from PsyNeuLink.Globals.Keywords import NAME, VARIABLE, FUNCTION, VALUE, PARAMS, TIME_SCALE, OBJECTIVE_MECHANISM, \
                                        INPUT_STATES, PROJECTIONS, WEIGHT, WEIGHTS, EXPONENT, EXPONENTS, SENDER, \
                                        MATRIX, DEFAULT_MATRIX, AUTO_ASSIGN_MATRIX, \
                                        OUTPUT_STATE, OUTPUT_STATES, LEARNING, CONTROL, MECHANISM, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList
from PsyNeuLink.Scheduling.TimeScale import TimeScale

ROLE = 'role'
MONITORED_OUTPUT_STATES = 'monitored_output_states'
MONITORED_OUTPUT_STATE_NAME_SUFFIX = '_Monitor'
OUTCOME = 'outcome'

# Indices for items in tuple format used for specifying monitored_output_states using weights and exponents
OUTPUT_STATE_INDEX = 0
WEIGHT_INDEX = 1
EXPONENT_INDEX = 2


# This is a convenience class that provides list of standard_output_state names in IDE
class OBJECTIVE_OUTPUT():
    """
    .. _ObjectiveMechanism_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `ObjectiveMechanism`:

    .. _OBJECTIVE_MECHANISM_OUTCOME

    *OUTCOME* : 1d np.array
        the value of the objective or "loss" function computed based on the
        ObjectiveMechanism's `function <ObjectiveMechanism.function>`

    """
    OUTCOME=OUTCOME


class ObjectiveMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ObjectiveMechanism(ProcessingMechanism_Base):
    """
    ObjectiveMechanism(               \
        monitored_output_states,             \
        input_states=None,            \
        function=LinearCombination,   \
        output_states=[OUTCOME],      \
        params=None,                  \
        name=None,                    \
        prefs=None)

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that evaluates the value(s)
    of one or more `OutputStates <OutputState>`.

    COMMENT:
        Description:
            ObjectiveMechanism is a subtype of the ProcessingMechanism Type of the Mechanism Category of the
                Component class
            Its function uses the LinearCombination Function to compare two input variables
            COMPARISON_OPERATION (functionParams) determines whether the comparison is subtractive or divisive
            The function returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
                as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

        Class attributes:
            + componentType (str): ObjectiveMechanism
            + classPreference (PreferenceSet): ObjectiveMechanism_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + ClassDefaults.variable (value):  None (must be specified using **input_states** and/or
                                               **monitored_output_states**)
            + paramClassDefaults (dict): {FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}

        Class methods:
            None

        MechanismRegistry:
            All instances of ObjectiveMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    monitored_output_states : List[`OutputState`, `Mechanism`, str, value, dict, `MonitoredOutputStatesOption`] or Dict
        specifies the OutputStates, the `value <OutputState.value>`\\s of which will be monitored, and evaluated by
        the ObjectiveMechanism's `function <ObjectiveMechanism>` (see `monitored_output_states
        <ObjectiveMechanism.monitored_output_states>` for details of specification).

    input_states :  List[InputState, value, str or dict] or Dict[] : default None
        specifies the names and/or formats to use for the values of the InputStates that receive the input from the
        OutputStates specified in the monitored_output_states** argument; if specified, there must be one for each item
        specified in the **monitored_output_states** argument.

    COMMENT:
    names: List[str]
        specifies the names to use for the input_states created for the list in
        `monitored_output_states <ObjectiveMechanism.monitor>`.  If specified,
        the number of items in the list must equal the number of items in `monitored_output_states`, and takes precedence
        over any names specified there.
    COMMENT

    function: CombinationFunction, ObjectiveFunction, function or method : default LinearCombination
        specifies the function used to evaluate the values listed in :keyword:`monitored_output_states`
        (see `function <LearningMechanism.function>` for details.

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [OUTCOME]
        specifies the OutputStates for the Mechanism;

    role: Optional[LEARNING, CONTROL]
        specifies if the ObjectiveMechanism is being used for learning or control (see `role` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default ObjectiveMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    COMMENT:
    default_variable : Optional[List[array] or 2d np.array]
    COMMENT

    monitored_output_states : ContentAddressableList[OutputState]
        determines the OutputStates, the `value <OutputState.value>`\\s of which are monitored, and evaluated by the
        ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  Each item in the list refers to an
        `OutputState` containing the value to be monitored, with a `MappingProjection` from it to the
        corresponding `InputState` listed in the `input_states <ObjectiveMechanism.input_states>` attribute.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains a weight and exponent associated with a corresponding InputState listed in the
        ObjectiveMechanism's `input_states <ObjectiveMechanism.input_states>` attribute;  these are used by its
        `function <ObjectiveMechanism.function>` to parametrize the contribution that the values of each of the
        OuputStates monitored by the ObjectiveMechanism makes to its output (see `ObjectiveMechanism_Function`)

    input_states : ContentAddressableList[InputState]
        contains the InputStates of the ObjectiveMechanism, each of which receives a `MappingProjection` from the
        OutputStates specified in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute.

    function : CombinationFunction, ObjectiveFunction, function, or method
        the function used to compare evaluate the values monitored by the ObjectiveMechanism.  The function can be
        any PsyNeuLink `CombinationFunction` or a Python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ObjectiveMechanism's variable (and its number of
        input_states), and returns a 1d array.

    role : None, LEARNING or CONTROL
        specifies whether the ObjectiveMechanism is used for learning in a Process or System (in conjunction with a
        `LearningMechanism`), or for control in a System (in conjunction with a `ControlMechanism <ControlMechanism>`).

    value : 1d np.array
        the output of the evaluation carried out by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.

    output_state : OutputState
        contains the `primary OutputState <OutputState_Primary>` of the ObjectiveMechanism; the default is
        its *OUTCOME* `OutputState <ObjectiveMechanism_Output>`, the value of which is equal to the
        `value <ObjectiveMechanism.value>` attribute of the ObjectiveMechanism.

    output_states : ContentAddressableList[OutputState]
        by default, contains only the *OUTCOME* (`primary <OutputState_Primary>`) OutputState of the ObjectiveMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *OUTCOME* `OutputState <ObjectiveMechanism_Output>`.

    name : str : default ObjectiveMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # ClassDefaults.variable = None;  Must be specified using either **input_states** or **monitored_output_states**
    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        variable = None

    # ObjectiveMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        MONITORED_OUTPUT_STATES: None,
        TIME_SCALE: TimeScale.TRIAL,
        FUNCTION: LinearCombination,
        })

    standard_output_states = standard_output_states.copy()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 monitored_output_states:tc.any(list, dict),
                 input_states=None,
                 function=LinearCombination,
                 output_states:tc.optional(tc.any(list, dict))=[OUTCOME],
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitored_output_states=monitored_output_states,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  function=function,
                                                  params=params)
        self._learning_role = None

        from PsyNeuLink.Components.States.OutputState import StandardOutputStates
        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        super().__init__(variable=None,
                         input_states=input_states,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):
        """Validate that if default_variable is specified the number of values matches the number of monitored_output_states

        """
        # NOTE 6/29/17: (CW)
        # This is a very questionable check. The problem is that TransferMechanism (if default_variable is passed as
        # None) expects variable to be initialized to ClassDefaults.variable ([[0]]) while ObjectiveMechanism expects
        # variable to be initialized to ClassDefaults.variable ([[0]]) AFTER this check has occurred. The problem is,
        # my solution to this has been to write (in each subclass of ProcessingMechanism) specific behavior on how to
        # react if both variable and size are None. This is fine but potentially cumbersome for future developers.
        # We should consider deleting this check entirely, and allowing ProcessingMechanism (or a further parent class)
        # to always set variable to ClassDefaults.variable if variable and size are both None.
        # IMPLEMENTATION NOTE:  use self.user_params (i.e., values specified in constructor)
        #                       since params have not yet been validated and so self.params is not yet available
        if variable is not None and len(variable) != len(self.user_params[MONITORED_OUTPUT_STATES]):
            raise ObjectiveMechanismError("The number of items specified for the default_variable arg ({}) of {} "
                                          "must match the number of items specified for its monitored_output_states arg ({})".
                                          format(len(variable), self.name, len(self.user_params[MONITORED_OUTPUT_STATES])))
        # MODIFIED 6/29/17 END

        return super()._validate_variable(variable=variable, context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate **role**, **monitored_output_states**, amd **input_states** arguments

        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if ROLE in target_set and target_set[ROLE] and not target_set[ROLE] in {LEARNING, CONTROL}:
            raise ObjectiveMechanismError("\'role\'arg ({}) of {} must be either \'LEARNING\' or \'CONTROL\'".
                                          format(target_set[ROLE], self.name))

        if (INPUT_STATES in target_set and
                    target_set[INPUT_STATES] is not None and
                not all(input_state is None for input_state in target_set[INPUT_STATES])):
            if MONITORED_OUTPUT_STATES in target_set:
                monitored_output_states = target_set[MONITORED_OUTPUT_STATES]
            elif hasattr(self, 'monitored_output_states'):
                monitored_output_states = self.monitored_output_states
            else:
                raise ObjectiveMechanismError("PROGRAM ERROR: monitored_output_states not instantiated as param or attirb")

            if len(target_set[INPUT_STATES]) != len(monitored_output_states):
                raise ObjectiveMechanismError("The number of items in the \'{}\'arg for {} ({}) "
                                              "must equal of the number in the \'{}\' arg ({})".
                                     format(INPUT_STATES,
                                            self.name,
                                            len(target_set[INPUT_STATES]),
                                            MONITORED_OUTPUT_STATES,
                                        len(target_set[MONITORED_OUTPUT_STATES])))

            #FIX: IS THIS HANDLED BY _instantiate_input_states??
            for state_spec in target_set[INPUT_STATES]:
                if not isinstance(state_spec, (str, InputState, Mechanism, dict)):
                    raise ObjectiveMechanismError("Item in \'{}\'arg for {} is not a "
                                                  "valid specificaton for an InputState".
                                                  format(INPUT_STATES,
                                                         self.name,
                                                         target_set[INPUT_STATES]))

        #region VALIDATE MONITORED VALUES
        # FIX: IS THE FOLLOWING STILL TRUE:
        # Note: this must be validated after OUTPUT_STATES (and therefore call to super._validate_params)
        #       as it can reference entries in that param
        if MONITORED_OUTPUT_STATES in target_set:
            try:
                if not target_set[MONITORED_OUTPUT_STATES] or target_set[MONITORED_OUTPUT_STATES] is NotImplemented:
                    pass
                # It is a MonitoredOutputStatesOption specification
                elif isinstance(target_set[MONITORED_OUTPUT_STATES], MonitoredOutputStatesOption):
                    # Put in a list (standard format for processing by _parse_monitored_output_states_list)
                    target_set[MONITORED_OUTPUT_STATES] = [target_set[MONITORED_OUTPUT_STATES]]
                # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
                else:
                    # Validate each item of MONITORED_OUTPUT_STATES
                    for item in target_set[MONITORED_OUTPUT_STATES]:
                        _parse_monitored_output_states(source=self, output_state_list=item, context=context)

            except KeyError:
                pass

    def _instantiate_input_states(self, monitored_output_states_specs=None, context=None):
        """Instantiate InputStates for each OutputState specified in monitored_output_states_specs

        Called by _add_monitored_output_states as well as during initialization
            (so must distinguish between initialization and adding to instantiated input_states)

        Parse specifications for **input_states**, using **monitored_output_states** where relevant and instantiate
        input_states.

        Instantiate or extend self.instance_defaults.variable to match number of InputStates.

        Update self.input_state and self.input_states.

        Call _instantiate_monitoring_projection() to instantiate MappingProjection to InputState
            if an OutputState has been specified.
        """

        # If call is for initialization
        if self.init_status is InitStatus.UNSET:
            # Pass self.input_states (containing specs from **input_states** arg of constructor)
            input_states = self.input_states
        else:
            # If initialized, don't pass self.input_states, as this is now a list of existing InputStates
            input_states = None
        monitored_output_states = monitored_output_states_specs or self.monitored_output_states
        input_state_dicts, output_state_dicts = self._instantiate_monitored_output_states(monitored_output_states=monitored_output_states,
                                                                                   input_states=input_states,
                                                                                   context=context)

       # For each InputState specified by monitored_output_states, add an item to self.variable
        #    - get specified constraint on variable and add item to self.variable
        self.instance_defaults.variable = self.instance_defaults.variable or []
        for i, input_state_dict in enumerate(input_state_dicts):
            self.instance_defaults.variable.append(input_state_dict[VARIABLE])

        # Instantiate InputStates corresponding to OutputStates specified in specified monitored_output_states
        instantiated_input_states = super()._instantiate_input_states(input_states=input_state_dicts, context=context)

        # Get any Projections specified in input_states arg, else set to default (AUTO_ASSIGN_MATRIX)
        # Only do this during initialization;  otherwise, self._input_states has actual states, not specifications.
        if self.init_status is InitStatus.UNSET:
            input_state_projection_specs = []
            for i, state in enumerate(self._input_states):
                input_state_projection_specs.append(state.params[PROJECTIONS] or [AUTO_ASSIGN_MATRIX])
        # FIX: END DIFF

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        output_states = [monitored_value[OUTPUT_STATE] for monitored_value in output_state_dicts]
        if output_states:
            _instantiate_monitoring_projections(owner=self,
                                                sender_list=output_states,
                                                receiver_list=instantiated_input_states,
                                                receiver_projection_specs=input_state_projection_specs,
                                                context=context)

    def _instantiate_monitored_output_states(self, monitored_output_states, input_states=None, context=None):
        """Parse monitored_value specs and instantiate monitored_output_states attribute

        Used by _instantiate_input_states and _add_monitored_output_states
            (so must distinguish between initialization and adding to instantiated input_states)
        monitored_output_states argument:
            list of any legal specification for **monitored_output_states** argument of the constructor;
        input_states argument:
            list of input_state specifications,
            specified in the **input_states** arg of the constructor, and used for default values
                not overridden by specifications monitored_output_states (e.g., weight and exponents)

        Returns list of tuples, each of which is a pair of InputState and OutputState specification dictionaries;
            input_state_dicts: specification for InputStates to be created
            output_state_dicts:  OutputStates to be monitored by the InputStates
        """

        # First parse for tuples to extract OutputStates, weights and exponents
        monitored_output_states_parsed = _parse_monitored_output_states(source=self,
                                                          output_state_list=monitored_output_states,
                                                          context=context)
        output_state_specs = [s[OUTPUT_STATE_INDEX] for s in monitored_output_states_parsed]
        monitored_value_weights = [w[WEIGHT_INDEX] for w in monitored_output_states_parsed]
        monitored_value_exponents = [e[EXPONENT_INDEX] for e in monitored_output_states_parsed]

        # Then, parse OutputState specifications
        output_state_dicts = []
        for value in output_state_specs:
            output_state_dict = {}
            output_state_spec = _parse_state_spec(owner=self,
                                                  state_type=OutputState,
                                                  state_spec=value)
            if isinstance(output_state_spec, dict):
                output_state_dict = output_state_spec
            elif isinstance(output_state_spec, State):
                output_state_dict[NAME] = output_state_spec.name
                output_state_dict[VARIABLE] = output_state_spec.instance_defaults.variable
                output_state_dict[VALUE] = output_state_spec.instance_defaults.variable
                # output_state_dict[PARAMS] = output_state_spec.params
            else:
                raise ObjectiveMechanismError("PROGRAM ERROR: call to State._parse_state_spec() for {} of {} "
                                              "should have returned dict or State, but returned {} instead ({})".
                                              format(OUTPUT_STATE,
                                                     self.name,
                                                     type(output_state_spec).__name__,
                                                     output_state_spec))
            output_state_dict[OUTPUT_STATE]=value
            output_state_dict[NAME] = output_state_dict[NAME] + MONITORED_OUTPUT_STATE_NAME_SUFFIX

            # Ignore any OutputStates from which an InputState of the ObjectiveMechanism already receives a Projection.
            # However, skip this step if this call is during initialization of the ObjectiveMechanism, because:
            #    - it should not have yet any InputStates instantiated
            #    - but can't test for self.input_states==None, since that might contain specifications from the
            #      input_states argument of the ObjectiveMechanism's constructor
            if not self.init_status is InitStatus.UNSET:
                if output_state_dict[OUTPUT_STATE] in self.monitored_output_states:
                    if any(any(projection.sender is output_state_dict[OUTPUT_STATE]
                           for projection in input_state.path_afferents)
                        for input_state in self.input_states):
                        continue

            output_state_dicts.append(output_state_dict)

        # INSTANTIATE InputState for ObjectiveMechanism CORRESPONDING TO EACH OutputState specified in monitored_output_states

        # If input_states were provided use those for specifications;
        #    otherwise use value of monitored_valued for each (to invoke a default assignment for each input_state)
        input_state_specs = input_states or [output_state_dict[VALUE] for output_state_dict in output_state_dicts]

        # Parse input_states into a state specification dict, passing output_state_specs as defaults
        input_state_dicts = []
        for i, input_state, output_state_dict in zip(range(len(input_state_specs)),
                                                     input_state_specs,
                                                     output_state_dicts):
            # Parse input_state to determine its specifications and assign values from output_state_specs
            #    to any missing specifications, including any projections requested and weights and exponents.
            input_state_dict = _parse_state_spec(self,
                                                 state_type=InputState,
                                                 state_spec=input_state,
                                                 name=output_state_dict[NAME],
                                                 value=output_state_dict[VALUE])
            input_state_dicts.append(input_state_dict)

        # For each InputState specified by monitored_output_states:
        #    - assign weights and exponents (if specified in monitored_value to params dict
        #      (otherwise, don't include param dict)
        for i, input_state_dict in enumerate(input_state_dicts):
            weight_and_exponent_params_dict = {}
            if monitored_value_weights[i] is not None:
                weight_and_exponent_params_dict[WEIGHT] = monitored_value_weights[i]
            if monitored_value_exponents[i] is not None:
                weight_and_exponent_params_dict[EXPONENT] = monitored_value_exponents[i]
            # If the input_state_dict already had a PARAMS entry, add WEIGHT and EXPONENT ENTRIES to it
            if input_state_dict[PARAMS] is not None:
                input_state_dict[PARAMS].update(weight_and_exponent_params_dict)
            # Otherwise, if any weights and/or exponents were specified, assign dict with them to PARAMS entry
            elif weight_and_exponent_params_dict:
                input_state_dict[PARAMS] = weight_and_exponent_params_dict


        output_states = [monitored_value[OUTPUT_STATE] for monitored_value in output_state_dicts]
        if output_states:
            if self.init_status is InitStatus.UNSET:
                # Call is during initialization -- vs. and addition of monitored_value(s) -- so assign output_states
                self.monitored_output_states = output_states
            else:
                # Call is to add monitored_value(s) -- so append output_states
                self.monitored_output_states.append(output_states)

        return input_state_dicts, output_state_dicts

    def add_monitored_output_states(self, monitored_output_states_specs, context=None):
        """Instantiate OutputStates to be monitored by ObjectiveMechanism

        Used by other objects to add a state or list of states to be monitored by ObjectiveMechanism.
        monitored_output_states_spec can be a Mechanism, OutputState, monitored_value tuple, or list with any of these.
        If item is a Mechanism, its primary OutputState is used.
        """
        monitored_output_states_specs = list(monitored_output_states_specs)

        # FIX: NEEDS TO RETURN output_states (?IN ADDITION TO input_states) SO THAT IF CALLED BY ControlMechanism THAT
        # FIX:  BELONGS TO A SYSTEM, THE ControlMechanism CAN CALL System._validate_monitored_state_in_system
        # FIX:  ON THE output_states ADDED
        return self._instantiate_input_states(monitored_output_states_specs=monitored_output_states_specs, context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Assign InputState weights and exponents to ObjectiveMechanism's function
        """
        super()._instantiate_attributes_after_function(context=context)
        self._instantiate_function_weights_and_exponents(context=context)

    def _instantiate_function_weights_and_exponents(self, context=None):
        """Assign weights and exponents to ObjectiveMechanism's function if it has those attributes

        For each, only make assignment if one or more entries in it has been assigned a value
        If any one value has been assigned, assign default value (1) to all other elements
        """
        DEFAULT_WEIGHT = 1
        DEFAULT_EXPONENT = 1

        weights = [input_state.weight for input_state in self.input_states]
        exponents = [input_state.exponent for input_state in self.input_states]

        if hasattr(self.function_object, WEIGHTS):
            if any(weight is not None for weight in weights):
                self.function_object.weights = [weight or DEFAULT_WEIGHT for weight in weights]
        if hasattr(self.function_object, EXPONENTS):
            if any(exponent is not None for exponent in exponents):
                self.function_object.exponents = [exponent or DEFAULT_EXPONENT for exponent in exponents]

    @property
    def monitored_output_states_weights_and_exponents(self):
        if hasattr(self.function_object, WEIGHTS) and self.function_object.weights is not None:
            weights = self.function_object.weights
        else:
            weights = [input_state.weight for input_state in self.input_states]
        if hasattr(self.function_object, EXPONENTS) and self.function_object.exponents is not None:
            exponents = self.function_object.exponents
        else:
            exponents = [input_state.exponent for input_state in self.input_states]
        return [(w,e) for w, e in zip(weights,exponents)]

    @monitored_output_states_weights_and_exponents.setter
    def monitored_output_states_weights_and_exponents(self, weights_and_exponents_tuples):

        weights = [w[0] for w in weights_and_exponents_tuples]
        exponents = [e[1] for e in weights_and_exponents_tuples]
        self._instantiate_weights_and_exponents(weights, exponents)


def _parse_monitored_output_states(source, output_state_list, mech=None, context=None):
    """Parses specifications in list and returns list of tuples: [(OutputState, exponent, weight)...]

    Serves the following purposes:

        Validates that each item of monitored_value arg is:
            * OutputState
            * Mechanism,
            * tuple: (OutputState/Mechanism, weight, exponent) tuple
            * dictionary: {MECHANISM:<Mechanism>, <OUTPUT_STATES:[<legal monitored_value spec>, ...]}
            * string
            * MonitoredOutpuStatesOption value

    Extract references to Mechanisms and/or OutputStates, exponents and weights from each tuple
    and assign each to its own list
    Assign None as default for unspecified weights and exponents, so that specification of these for
       InputStates can be used (see ObjectiveMechanism._instantiate_input_states)

    Called by:
        self._validate_params()
        self._instantiate_input_states()
        self._instantiate_monitored_output_states()
        ControlMechanism._instantiate_objective_mechanism()
        System._get_monitored_output_states_for_system()

        ERROR MESSAGE:
        raise ObjectiveMechanismError("Specification of OutputState to be monitored for {} ({}) must be an "
                                      "OutputState, the name or a specification dictionary for one, "
                                      "a Mechanism, a (Mechanism, weight, exponent) tuple, or a value of "
                                      "MonitoredOutputStatesOption".
                                      format(component.name, state_spec))

    """

    if not isinstance(output_state_list, list):
        output_state_list = [output_state_list]

    # output_states = np.array([None] * len(output_state_list))
    # weights = output_states.copy()
    # exponents = weights.copy()
    output_states = []
    weights = []
    exponents = []
    DEFAULT_WEIGHT = None
    DEFAULT_EXPONENT = None

    for item in output_state_list:

        # Specification is an OutputState on its own, so use it and assign default weight and exponent
        if isinstance(item, OutputState):
            output_states.append(item)
            weights.append(DEFAULT_WEIGHT)
            exponents.append(DEFAULT_EXPONENT)

        # Specification is a Mechanism, so use its primary OutputState and assign default weight and exponent
        elif isinstance(item, Mechanism):
            output_states.append(item.output_state)
            weights.append(DEFAULT_WEIGHT)
            exponents.append(DEFAULT_EXPONENT)

        # Specification is a string so see if it is the name of an OutputState
        elif isinstance(item, str):
            if mech is not None:
                # If mech was specified, interpret str as the name of one of its OutputStates
                try:
                    output_states.append(mech.output_states[item])
                except KeyError:
                    raise ObjectiveMechanismError("{} does not have an OutputState named {} "
                                                  "(used to specify it in monitored_output_states for {})".
                                                  format(mech.name, item, source.name))
                except AttributeError:
                    raise ObjectiveMechanismError("MECHANISM entry ({}) of monitored_output_states specification dictionary "
                                                  "for {} does not appear to have an output_states attribute".
                                                  format(mech.name, source.name))
            else:
                # Specification is an "unbound" string, so pass it along
                #    (note:  strings can be used as "placemarkers" for InputStates that will be resolved later,
                #            e.g., for TARGET InputState of a ObjectiveMechanism)
                output_states.append(item)
            weights.append(DEFAULT_WEIGHT)
            exponents.append(DEFAULT_EXPONENT)

        # Specification is a tuple so parse
        elif isinstance(item, tuple):
            #  Check that it has three items:  (monitored_value specification, weight, exponent)
            if len(item) != 3:
                raise ObjectiveMechanismError("Tuple {} used for monitored_value specification in {} "
                                     "has {} items;  it should be 3".
                                     format(item, source.name, len(item)))

            # Recursively call _parse_monitored_output_states to parse the first item of the tuple
            # (index with 0 since parse returns a list, but should only be one item in it)
            output_state_tuple = _parse_monitored_output_states(source,
                                                         item[OUTPUT_STATE_INDEX],
                                                         mech=mech, context=context)[0]

            # Use OutputState in tuple returned from parse
            output_state = output_state_tuple[OUTPUT_STATE_INDEX]

            # If output_state is a string,
            from PsyNeuLink.Components.System import System
            if isinstance(output_state, str):
                if not isinstance(source, System):
                    raise ObjectiveMechanismError("A string was used to specify an OutputState (presumably its name: "
                                                  "\'{}\') to be monitored by the ObjectiveMechanism for {}; the name "
                                                  "of an OutputState can be used to specify it only in the "
                                                  "\'monitor_for_control\' argument of a System, but NOT in the "
                                                  "\'objective_mechanism\' argument of a ControlMechanism nor in the "
                                                  "\'monitored_output_states\' argument of an ObjectiveMechanism".
                                                  format(output_state, source.name))

            elif not isinstance(output_state, (OutputState, Mechanism)):
                raise ObjectiveMechanismError("PROGRAM ERROR: parse_monitored_output_states() returned a tuple for {}, "
                                              "the first item of which ({}) is not an OutputState, the name of one, "
                                              "or Mechanism".format(source.name, output_state))
            output_states.append(output_state)

            # Use weight and exponent if returned from parse;  otherwise use what was in item
            weight = output_state_tuple[WEIGHT_INDEX] or item[WEIGHT_INDEX]
            exponent = output_state_tuple[EXPONENT_INDEX] or item[EXPONENT_INDEX]

            if weight and not isinstance(weight, numbers.Number):
                raise ObjectiveMechanismError("Specification of the weight ({}) in tuple for {} of {} "
                                     "must be a number".
                                     format(weight, item[OUTPUT_STATE_INDEX].name, source.name))
            weights.append(weight)

            if exponent and not isinstance(exponent, numbers.Number):
                raise ObjectiveMechanismError("Specification of the exponent ({}) in tuple for {} of {} "
                                     "must be a number".
                                     format(exponent, item[OUTPUT_STATE_INDEX].name, source.name))
            exponents.append(exponent)

        # Specification is a dictionary, so parse
        elif isinstance(item, dict):
            if not MECHANISM in item:
                raise ObjectiveMechanismError("Specification dictionary used for monitored_output_states in {} "
                                              "is missing its MECHANISM entry".format(source.name))
            mech = item[MECHANISM]

            if OUTPUT_STATES in item:
                # Get list of OutputState specifications and append to output_states by
                #    recursively calling _parse_monitored_output_states to parse the list of OutputState entries
                #    (index with 0 since parse returns a list, but should only be one item in it)
                output_state_tuples = _parse_monitored_output_states(source=source,
                                                              output_state_list=item[OUTPUT_STATES],
                                                              mech=mech,
                                                              context=context)
                output_states.extend([output_state_tuple[OUTPUT_STATE_INDEX]
                                      for output_state_tuple in output_state_tuples])
                weights.extend([output_state_tuple[WEIGHT_INDEX]
                                      for output_state_tuple in output_state_tuples])
                exponents.extend([output_state_tuple[EXPONENT_INDEX]
                                      for output_state_tuple in output_state_tuples])

            else:
                # MECHANISM was the only entry in the dict,
                #     so use it as a specification for the primary OutputState of the Mechanism
                output_states.append(mech.output_state)
                weights.append(DEFAULT_WEIGHT)
                exponents.append(DEFAULT_EXPONENT)

        elif isinstance(item, MonitoredOutputStatesOption):
            output_states.append(item)
            weights.append(DEFAULT_WEIGHT)
            exponents.append(DEFAULT_EXPONENT)

        else:
            raise ObjectiveMechanismError("Unrecognized specification for monitor_value ({}) in {}".
                                          format(item, source.name))

    if not (len(output_states)==len(weights)==len(exponents)):
        raise ObjectiveMechanismError("PROGRAM ERROR: The lengths of the lists of OutputStates ({}), weights ({}) and "
                                      "exponents ({}) are not equal in the monitored_output_states specification for {}".
                                      format(len(output_states), len(weights), len(exponents), source.name))

    return list(zip(output_states, weights, exponents))

def _objective_mechanism_role(mech, role):
    if isinstance(mech, ObjectiveMechanism):
        if mech._role is role:
            return True
        else:
            return False
    else:
        return False

# IMPLEMENTATION NOTE: THIS IS A PLACEMARKER FOR A METHOD TO BE IMPLEMENTED IN THE Composition CLASS
#                      ??MAYBE INTEGRATE INTO State MODULE (IN _instantate_state)
@tc.typecheck
def _instantiate_monitoring_projections(owner,
                                        sender_list:tc.any(list, ContentAddressableList),
                                        receiver_list:tc.any(list, ContentAddressableList),
                                        receiver_projection_specs:tc.optional(list)=None,
                                        context=None):

    from PsyNeuLink.Components.States.OutputState import OutputState
    from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

    receiver_projection_specs = receiver_projection_specs or [DEFAULT_MATRIX] * len(sender_list)

    if len(sender_list) != len(receiver_list):
        raise ObjectiveMechanismError("PROGRAM ERROR: Number of senders ({}) does not equal number of receivers ({}) "
                                     "in call to instantiate monitoring projections for {}".
                                     format(len(sender_list), len(receiver_list), owner.name))

    if len(receiver_projection_specs) != len(receiver_list):
        raise ObjectiveMechanismError("PROGRAM ERROR: Number of projection specs ({}) "
                                     "does not equal number of receivers ({}) "
                                     "in call to instantiate monitoring projections for {}".
                                     format(len(receiver_projection_specs), len(receiver_list), owner.name))

    # Instantiate InputState with Projection from OutputState specified by sender
    for sender, receiver, recvr_projs in zip(sender_list, receiver_list, receiver_projection_specs):

        # IMPLEMENTATION NOTE:  If there is more than one Projection specified for a receiver, only the 1st is used;
        #                           (there should only be one if a 2-item tuple was used to specify the InputState,
        #                            however other forms of specifications could produce more)
        if len(recvr_projs) > 1 and owner.verbosePref:
            warnings.warn("{} projections were specified for InputState ({}) of {} ;"
                          "only the first ({}) will be used".
                          format(len(recvr_projs), receiver.name, owner.name))
        projection_spec = recvr_projs[0]


        # IMPLEMENTATION NOTE:  This may not handle situations properly in which the OutputState is specified
        #                           by a 2-item tuple (i.e., with a Projection specification as its second item)
        if isinstance(sender, OutputState):
            # Projection has been specified for receiver and initialization begun, so call deferred_init()
            if receiver.path_afferents:
                if not receiver.path_afferents[0].init_status is InitStatus.DEFERRED_INITIALIZATION:
                    raise ObjectiveMechanismError("PROGRAM ERROR: {} of {} already has an afferent projection "
                                                  "implemented and initialized ({})".
                                                  format(receiver.name, owner.name, receiver.aferents[0].name))
                if not receiver.path_afferents[0].function_params[MATRIX] is projection_spec:
                    raise ObjectiveMechanismError("PROGRAM ERROR: Projection specification for {} of {} ({}) "
                                                  "does not match matrix already assigned ({})".
                                                  format(receiver.name,
                                                         owner.name,
                                                         projection_spec,
                                                         receiver.path_afferents[0].function_params[MATRIX]))
                receiver.path_afferents[0].init_args[SENDER] = sender
                receiver.path_afferents[0]._deferred_init()
            else:
                MappingProjection(sender=sender,
                                  receiver=receiver,
                                  matrix=projection_spec,
                                  name = sender.name + ' monitor')
