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
ObjectiveMechanism's *OUTCOME* (`primary <OutputState_Primary>`) OutputState.  ObjectiveMechanisms are typically used
closely with (and often created automatically by) `AdaptiveMechanisms <AdaptiveMechanism>`.

.. _ObjectiveMechanism_Creation:

Creating an ObjectiveMechanism
------------------------------

ObjectiveMechanisms are often created automatically when other PsyNeuLink components are created (in particular,
AdaptiveMechanisms, such as `LearningMechanisms <LearningMechanism_Creation>` and
`ControlMechanisms <ControlMechanism_Creation>`).  An ObjectiveMechanism can also be created directly by calling its
constructor.  The primary attribute used to define an ObjectiveMechanism is its `monitor <ObjectiveMechanism.monitor>`
attribute, that is specified using the corresponding argument of its constructor as described below.

.. _ObjectiveMechanism_Monitor:

*Monitor*
~~~~~~~~~

COMMENT:
FOR DEVELOPERS:
    The monitor argument is in effect an alias to the input_states argument
    of the constructor for a Mechanism;  it is simply assigned to input_state in the __init__ method
    and the specifications are handled by an override of the Mechanism's _instantiate_input_states method.
    The monitor property returns the OutputStates that project to the Mechanism's InputStates
COMMENT

The **monitor** argument of an ObjectiveMechanism's constructor specifies the `OutputStates <OutputState>` it monitors.
This takes the place of the **input_states** argument used by most other forms of `Mechanism <Mechanism>`, and is used
by the ObjectiveMechanism to create an `InputState` for each OutputState it monitors, along with a `MappingProjection`
from the OutputState to that InputState.  The **monitor** argument takes a list of items that can
include any of the `forms of specification <InputState_Specification>` used in a standard **input_states** argument.
For the **monitor** argument, this is usually a list of OutputStates to be monitored.  However, as with a standard
**input_states** argument, the **monitor** argument can include Mechanisms (in which case their `primary Outputstate
<OutputState_Primary>` is used) or the `InputState(s) <InputState>` of other Mechanisms (in which case the
ObjectiveMechanism will be assigned Projections from all of the OutputStates that project to the specified InputState
-- that is, it will `shadow their inputs <InputState_Shadow_Inputs>`). Items in the *monitor* argument can also be
used to specify attributes of the InputState and/or MappingProjection(s) to it, that the ObjectiveMechanism creates to
monitor the specified OutputState.  In general, the `value <OutputState.value>` of each specified OutputState determines
the format of the `variable <InputState.variable>` of the InputState that is created for it by the ObjectiveMechanism.
However, this can be overridden using the ObjectiveMechanism's `default_variable <ObjectiveMechanism.default_variable>`
or `size <ObjectiveMechanism.size>` attributes (see `Mechanism InputState specification
<Mechanism_InputState_Specification>`), or by specifying a Projection from the OutputState to the InputState (see
`Input Source Specification <InputState_Projection_Source_Specification>`). If an item in the
**monitor** argument specifies an InputState for the ObjectiveMechanism, but not the OutputState to
be monitored, the InputState is created but will be ignored until an OutputState (and MappingProjection from it) are
specified for that InputState.

COMMENT:
FIX 8/27/19 [JDC]:
ADD DISCUSSION OF monitor_weights_and_exponents ARGUMENT HERE
.. _ObjectiveMechanism_Monitor_Weights_and_Exponents:
COMMENT

COMMENT:
Note that some forms of specification may depend on specifications made for the OutputState referenced, the Mechanism
to which it belongs, and/or the Process or System to which that Mechanism belongs. These interactions (and the
precedence afforded to each) are described below.

  * **OutputState** -- a reference to the `OutputState <OutputState>` of a Mechanism;  this creates an InputState
    with a `variable <InputState.variable>` that matches the format of the `value <OutputState.value>` of the
    specified OutputState, and a `MappingProjection` between them using an *IDENTITY_MATRIX*.

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

TBI FOR COMPOSITION:
The OutputStates monitored by the ObjectiveMechanism are listed in its `monitor <ObjectiveMechanism.monitor>`
attribute.  When an ObjectiveMechanism is created by a `ControlMechanism`, or a `System` for its `controller
<System.controller>`, these may pass a set of OutputStates to be monitored to the ObjectiveMechanism.  A
ControlMechanism passes OutputState specifications listed in its **objective_mechanism** argument (see
`ControlMechanism_ObjectiveMechanism`), and a System passes any listed in its **monitor_for_control** argument (see
`System_Control_Specification`).
COMMENT

COMMENT:
FIX 8/27/19 [JDC]:
GENERALIZE TO ModulatoryMechanism
COMMENT

The OutputStates monitored by the ObjectiveMechanism are listed in its `monitor <ObjectiveMechanism.monitor>` attribute.
When an ObjectiveMechanism is created by a `ControlMechanism`, these may pass a set of OutputStates to be monitored to
the ObjectiveMechanism.  A ControlMechanism passes OutputState specifications listed in its **objective_mechanism**
argument (see `ControlMechanism_ObjectiveMechanism`).

.. _ObjectiveMechanism_Structure:


Structure
---------

.. _ObjectiveMechanism_Input:

*Input*
~~~~~~~

An ObjectiveMechanism has one `InputState <InputState>` for each of the OutputStates specified in its
**monitor** argument (see `ObjectiveMechanism_Monitor`). Each InputState receives a `MappingProjection` from the
corresponding OutputState, the values of which are used by the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` to generate the value of its *OUTCOME* `OutputState
<ObjectiveMechanism_Output>`.  The InputStates are listed in the ObjectiveMechanism's `input_states
<ObjectiveMechanism.input_states>` attribute, and the monitored OutputStates from which they receive projections are
listed in the same order its `monitor <ObjectiveMechanism.monitor>` attribute.

COMMENT:
  FIX: Shadowing inputs may generate an exception to this, if the shadowed InputState receives more than one projection.
       In that case, there will be more than on OutputState listed in monitor for the corresponding input_state
COMMENT

By default, the format of the `variable <InputState.variable>` for each InputState is determined by the `value
<OutputState.value>` of the monitored OutputState(s) to which it corresponds.  However, if either the
**default_variable** or **size** argument is specified in an Objective Mechanism's constructor, or a `variable
<InputState.variable>` is `specified for an InputState <InputState_Specification>` for one or more of the items in
its **monitor** argument, then that is used as the format for the corresponding InputState(s).  This can be used to
transform the `value <OutputState.value>` of a monitored OutputState into different form for the `variable
<InputState.variable>` of the InputState (see the `first example <ObjectiveMechanism_Monitor_Examples>` below).

If the weight and/or exponent is specified for ay item in the **monitor** argument of the ObjectiveMechanism's
constructor, it is assigned to the corresponding InputState.  If the ObjectiveMechanism's `function
<ObjectiveMechanism.function>` implements a weights and/or exponents attribute, the values specified is assigned to
the corresponding attribute, and applied to the `value <InputState.value>` of the InputState before it is combined
with that of the other InputStates to generate the ObjectiveMechanism's `output <ObjectiveMechanism_Output>`.


.. _ObjectiveMechanism_Function:

*Function*
~~~~~~~~~~

The ObjectiveMechanism's `function <ObjectiveMechanism.function>` uses the values of its `input_states
<ObjectiveMechanism.input_states>` to compute an `objective (or "loss") function
<https://en.wikipedia.org/wiki/Loss_function>`_, that is assigned as the value of its *OUTCOME* `OutputState
<ObjectiveMechanism_Output>`.  By default, it uses a `LinearCombination` function to sum the values of the values of
the items in its `variable <ObjectiveMechanism.variable>`. However, by assigning values to the 'weight
<InputState.weight>` and/or 'exponent <InputState.exponent>` attributes of the corresponding InputStates,
it can be configured to calculate differences, ratios,  etc. (see `example
<ObjectiveMechanism_Weights_and_Exponents_Example>` below).  The `function <ObjectiveMechanism.function>`  can also
be replaced with any `CombinationFunction`, or any python function that takes an 2d array as its input (with a number
of items in axis 0 equal to the number of the ObjectiveMechanism's InputStates), and generates a 1d array as its result.
If it implements :keyword:`weight` and/or :keyword:`exponent` attributes, those are assigned from `weight
<InputState.weight>` and `exponent <InputState.exponent>` attributes of its `input_states
<ObjectiveMechanism.input_states>` (also listed in the `monitor_weights_and_exponents
<ObjectiveMechanism.monitor_weights_and_exponents>` attribute);  otherwise, they are ignored.

.. _ObjectiveMechanism_Output:

*Output*
~~~~~~~~

The `primary OutputState <OutputState_Primary>` of an Objective mechanism is a 1d array, named *OUTCOME*, that is the
result of its `function <ObjectiveMechanism.function>` (as described above).


.. _ObjectiveMechanism_Execution:

Execution
---------

When an ObjectiveMechanism is executed, it updates its input_states with the values of the OutputStates listed in
its `monitor <ObjectiveMechanism.monitor>` attribute, and then uses its `function <ObjectiveMechanism.function>` to
evaluate these.  The result is assigned as to its `value <ObjectiveMechanism.value>` attribute as the value of its
*OUTCOME* (`primary <OutputState_Primary>`) OutputState.

.. _ObjectiveMechanism_Examples:

Examples
--------

.. _ObjectiveMechanism_Monitor_Examples:

*Specifying* the **variable** for the InputStates of an ObjectiveMechanism

This can be useful in some situations, and there are several ways it can be done. For example, for
`Reinforcement Learning <Reinforcement>`, an ObjectiveMechanism is used to monitor the rewards by an action selection
Mechanism (and used by a LearningMechanism to improve those predictions).  However, whereas the action selection
Mechanism generates a vector indicating the reward predicted by the selected action, the ObjectiveMechanism for RL
simply needs to know the magnitude of the reward predicted, irrespective of the action taken;  that is, it
simply requires a single scalar value indicating the magnitude of the predicted reward.  Thus, the vector of
action-related reward predictions needs to be condensed to a single predicted value for the ObjectiveMechanism.  This
can be accomplished in several ways, that are illustrated in the examples below.  In the first example,
a `TransferMechanism` with the `SoftMax` function (and the `PROB <Softmax.PROB>` as its output format) to select
an action and represent its reward prediction.  This generates a vector with a single non-zero value, which designates
the predicted reward for the selected action.  Because the output is a vector, by default the InputState of the
ObjectiveMechanism created to monitor it will also be a vector.  However, the ObjectiveMechanism requires this to be
a single value, that it can compare with the value of the reward Mechanism (monitoring the feedback provided by the
environment).  In the example below, this is accomplished by using `default_variable` in the constructor of the
ObjectiveMechanism to force the InputState for the ObjectiveMechanism to have a single value::

    >>> import psyneulink as pnl
    >>> my_action_select_mech = pnl.TransferMechanism(default_variable=[0, 0, 0],
    ...                                               function=pnl.SoftMax(output=pnl.PROB),
    ...                                               name='Action Selection Mech')

    >>> my_reward_mech = pnl.TransferMechanism(default_variable=[0],
    ...                                        name='Reward Mech')

    >>> my_objective_mech = pnl.ObjectiveMechanism(default_variable=[[0],[0]],
    ...                                            monitor=[my_action_select_mech, my_reward_mech])

Note that the OutputStates for the ``my_action_selection`` and ``my_reward_mech`` are specified
in `monitor`.  If that were the only specification, the InputState created for ``my_action_select_mech``
would be a vector of length 3.  This is overridden by specifying `default_variable` as an array with two
single-value arrays (one corresponding to ``my_action_select_mech`` and the other to ``my_reward_mech``).  This forces
the InputState for ``my_action_select_mech`` to have only a single element which, in turn, will cause a
MappingProjection to be created from  ``my_action_select_mech`` to the ObjectiveMechanism's InputState using a
`FULL_CONNECTIVITY_MATRIX` (the one used for `AUTO_ASSIGN_MATRIX` when the sender and receiver have values of
different lengths).  This produces the desired effect, since the action selected is the only non-zero value in the
output of ``my_action_select_mech``, and so the `FULL_CONNECTIVITY_MATRIX` will combine it with zeros (the other values
in the vector), and so its value will be assigned as the value of the corresponding InputState in the
ObjectiveMechanism.

An alternative would be to explicitly specify the `variable <InputState.variable>` attribute for the InputState created
for ``my_action_select_mech`` using a `InputState specification dictionary <InputState_Specification_Dictionary>` in
the **monitor** argument of ``my_objective_mech``, as follows::

    >>> my_objective_mech = pnl.ObjectiveMechanism(monitor=[{pnl.MECHANISM: my_action_select_mech,
    ...                                                      pnl.VARIABLE: [0]},
    ...                                                      my_reward_mech])

Note that the *VARIABLE* entry here specifies the `variable <InputState.variable>` for the InputState of the
ObjectiveMechanism created to receive a Projection from ``my_action_select_mech``, and not ``my_action_select_mech``
itself (see `ObjectiveMechanism_Input` for a full explanation).

.. _ObjectiveMechanism_Projection_Example:

Another way to specify the `variable <InputState.variable>` for the InputState of an ObjectiveMechanism is to
specify the Projections it receives from the OutputState it monitors.  The following example uses a `tuple
specification <InputState_Tuple_Specification:>` to assign the matrix for the MappingProjection from
``my_action_select_mech`` to the corresponding InputState of ``my_objective_mech``::

    >>> import numpy as np
    >>> my_objective_mech = pnl.ObjectiveMechanism(monitor=[(my_action_select_mech, np.ones((3,1))), my_reward_mech])

Since the matrix specified has three rows (for its inputs) and one col (for the output), it will take the length three
vector provided as the output of ``my_action_select_mech`` and combine its elements into a single value that is
provided to the InputState of the ObjectiveMechanism.

A `Connection tuple <InputState_Tuple_Specification>` could also be used to specify the matrix, but this would require
that additional entries (for the weight and exponent of the InputState) which, in this case, are not necessary (see
`example <ObjectiveMechanism_Tuple_Specification_Example>` below for how these can be used).

.. _ObjectiveMechanism_Weights_and_Exponents_Example:

By default, an ObjectiveMechanism simply adds the values received by each of its InputStates to generate its output.
However, this too can be customized in a variety of ways.

.. _ObjectiveMechanism_Tuple_Specification_Example:

The simplest way is to assign values to the `weight <InputState.weight>` and/or `exponent <InputState.exponent>`
attributes of its InputStates.  This can be done by placing them in a `tuple specification
<InputState_Tuple_Specification>` for the OutputState that provides value for the InputState. In the example
below, the ObjectiveMechanism used in the previous example is further customized to subtract the value of the action
selected from the value of the reward::

    >>> my_objective_mech = pnl.ObjectiveMechanism(default_variable = [[0],[0]],
    ...                                            monitor = [(my_action_select_mech, -1, 1), my_reward_mech])

This specifies that ``my_action_select_mech`` should be assigned a weight of -1 and an exponent of 1 when it is
submitted to the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  Notice that the exponent had to be
included, even though it is the default value;  when a tuple is used, the weight and exponent values must both be
specified.  Notice also that ``my_reward_mech`` does not use a tuple, so it will be assigned defaults for both the
weight and exponent parameters.

.. _ObjectiveMechanism_Multiple_OutputStates_Example:

An ObjectiveMechanism can also be configured to monitor multiple OutputStates of the same Mechanism.  In the following
example, an ObjectiveMechanism is configured to calculate the reward rate for a `DDM` Mechanism, by specifying
OutputStates for the DDM that report its response time and accuracy::

    >>> my_decision_mech = pnl.DDM(output_states=[pnl.RESPONSE_TIME,
    ...                                           pnl.PROBABILITY_UPPER_THRESHOLD])

    >>> my_objective_mech = pnl.ObjectiveMechanism(monitor=[
    ...                                              my_reward_mech,
    ...                                              my_decision_mech.output_states[pnl.PROBABILITY_UPPER_THRESHOLD],
    ...                                              (my_decision_mech.output_states[pnl.RESPONSE_TIME], 1, -1)])

This specifies that the ObjectiveMechanism should multiply the `value <OutputState.value>` of ``my_reward_mech``'s
`primary OutputState <OutputState_Primary>` by the `value <OutpuState.value>` of ``my_decision_mech``'s
*PROBABILITY_UPPER_THRESHOLD*, and divide the result by ``my_decision_mech``'s *RESPONSE_TIME* `value
<OutputState.value>`.  The two OutputStates of ``my_decision_mech`` are referenced as items in the `output_states
<Mechanism_Base.output_states>` list of ``my_decision_mech``.  However, a `2-item (State name, Mechanism) tuple
<InputState_State_Mechanism_Tuple>` can be used to reference them more simply, as follows::

    >>> my_objective_mech = pnl.ObjectiveMechanism(monitor=[
    ...                                           my_reward_mech,
    ...                                           (pnl.PROBABILITY_UPPER_THRESHOLD, my_decision_mech),
    ...                                           ((pnl.RESPONSE_TIME, my_decision_mech), 1, -1)])


*Customizing the ObjectiveMechanism's function*

In the examples above, the weights and exponents assigned to the InputStates are passed to the ObjectiveMechanism's
`function <ObjectiveMechanism.function>` for use in combining their values.  The same can be accomplished by
specifying the relevant parameter(s) of the function itself, as in the following example::

    >>> my_objective_mech = pnl.ObjectiveMechanism(default_variable = [[0],[0]],
    ...                                            monitor = [my_action_select_mech, my_reward_mech],
    ...                                            function=pnl.LinearCombination(weights=[[-1], [1]]))

Here, the `weights <LinearCombination.weights>` parameter of the `LinearCombination` function is specified directly,
with two values [-1] and [1] corresponding to the two items in `monitor` (and `default_variable`).
This will multiply the value from ``my_action_select_mech`` by -1 before adding it to (and thus subtracting it from)
the value of ``my_reward_mech``.  Notice that the weight for ``my_reward_mech`` had to be specified, even though it
is using the default value (1);  whenever a weight and/or exponent parameter is specified, there must be an entry for
every item of the function's variable.  However, the `exponents <LinearCombination.exponents>` did not need to be
specified, as it is not used.  However it, and the `operation <LinearCombination.operation>` parameters of
`LinearCombination` can also be used to multiply and divide quantities.

COMMENT:
**ADD DISCUSSION OF Projection WEIGHTS AND EXPONENTS ONCE THEY ARE IMPLEMENTED FOR HIERARCHICAL OPERATIONS
COMMENT

.. _ObjectiveMechanism_Class_Reference:

Class Reference
---------------

"""

import itertools
import warnings
import typecheck as tc
import numpy as np

from collections.abc import Iterable
from collections import namedtuple

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.states.outputstate import OutputState, PRIMARY, standard_output_states
from psyneulink.core.components.states.inputstate import InputState, INPUT_STATE
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    CONTEXT, CONTROL, EXPONENT, EXPONENTS, FUNCTION, LEARNING, MATRIX, NAME, \
    OBJECTIVE_MECHANISM, OUTCOME, PARAMS, PROJECTION, PROJECTIONS, STATE_TYPE, VARIABLE, WEIGHT, WEIGHTS, \
    kwPreferenceSetName
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList, type_match

__all__ = [
    'DEFAULT_MONITORED_STATE_WEIGHT', 'DEFAULT_MONITORED_STATE_EXPONENT', 'DEFAULT_MONITORED_STATE_MATRIX',
    'MONITOR', 'MONITOR_SUFFIX', 'MONITORED_OUTPUT_STATE_NAME_SUFFIX', 'MONITORED_OUTPUT_STATES',
    'OBJECTIVE_OUTPUT', 'ObjectiveMechanism', 'ObjectiveMechanismError', 'OUTCOME', 'ROLE'
]

ROLE = 'role'
MONITOR = 'monitor'
MONITOR_SUFFIX = '_Monitor'
MONITORED_OUTPUT_STATES = 'monitored_output_states'
MONITORED_OUTPUT_STATE_NAME_SUFFIX = '_Monitor'

DEFAULT_MONITORED_STATE_WEIGHT = None
DEFAULT_MONITORED_STATE_EXPONENT = None
DEFAULT_MONITORED_STATE_MATRIX = None

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
    # monitor is an alias to input_states argument, which can still be used in a spec dict
    """
    ObjectiveMechanism(               \
        monitor,                      \
        default_variable,             \
        size,                         \
        function=LinearCombination,   \
        output_states=OUTCOME,        \
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
            The function returns an array with the Hadamard (element-wise) difference/quotient of target vs. sample,
                as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

        Class attributes:
            + componentType (str): ObjectiveMechanism
            + classPreference (PreferenceSet): ObjectiveMechanism_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + class_defaults.variable (value):  None (must be specified using **input_states** and/or **monitor**)
            + paramClassDefaults (dict): {FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}

        Class methods:
            None

        MechanismRegistry:
            All instances of ObjectiveMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    monitor : List[`OutputState`, 'InputState`, `Mechanism`, str, value, dict, `MonitoredOutputStatesOption`] or dict
        specifies the OutputStates, the `values <OutputState.value>` of which will be monitored, and evaluated by
        `function <ObjectiveMechanism.function>` (see `ObjectiveMechanism_Monitor` for details of specification).

    default_variable : number, list or np.ndarray : default monitor
        specifies the format of the `variable <ObjectiveMechanism.variable>` for the `InputStates` of the
        ObjectiveMechanism (see `Mechanism InputState specification <Mechanism_InputState_Specification>` for details).

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    COMMENT:
    input_states :  List[InputState, value, str or dict] or Dict[] : default None
        specifies the names and/or formats to use for the values of the InputStates that receive the input from the
        OutputStates specified in the monitor** argument; if specified, there must be one for each item
        specified in the **monitor** argument.
    COMMENT

    function: CombinationFunction, ObjectiveFunction, function or method : default LinearCombination
        specifies the function used to evaluate the values listed in `monitor` <ObjectiveMechanism.monitor>`
        (see `function <ObjectiveMechanism.function>` for details).

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [OUTCOME]
        specifies the OutputStates for the Mechanism;

    role: Optional[LEARNING, CONTROL]
        specifies if the ObjectiveMechanism is being used for learning or control (see `role` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <ObjectiveMechanism.name>`
        specifies the name of the ObjectiveMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ObjectiveMechanism; see `prefs <ObjectiveMechanism.prefs>` for details.


    Attributes
    ----------

    variable : 2d ndarray : default array of values of OutputStates in monitor_output_states
        the input to Mechanism's `function <TransferMechanism.function>`.

    monitor : ContentAddressableList[OutputState]
        determines the OutputStates, the `values <OutputState.value>` of which are monitored, and evaluated by the
        ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  Each item in the list refers to an
        `OutputState` containing the value to be monitored, with a `MappingProjection` from it to an
        corresponding `InputState` listed in the `input_states <ObjectiveMechanism.input_states>` attribute

        .. note::
           If any of the ObjectiveMechanism's `input_states <ObjectiveMechanism.input_states>` were specified to `shadow
           the InputState <InputState_Shadow_Inputs>` of another Mechanism, and any of those shadowed InputStates
           receives more than one `Projection`, then the list of monitored OutputStates in `monitor` will be longer
           than the list of the ObjectiveMechanism's `input_states <ObjectiveMechanisms.input_states>`.

    monitor_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains a weight and exponent associated with a corresponding InputState listed in the
        ObjectiveMechanism's `input_states <ObjectiveMechanism.input_states>` attribute;  these are used by its
        `function <ObjectiveMechanism.function>` to parametrize the contribution that the values of each of the
        OuputStates monitored by the ObjectiveMechanism makes to its output (see `ObjectiveMechanism_Function`)

    input_states : ContentAddressableList[InputState]
        contains the InputStates of the ObjectiveMechanism, each of which receives a `MappingProjection` from the
        OutputStates specified in its `monitor <ObjectiveMechanism.monitor>` attribute.

    function : CombinationFunction, ObjectiveFunction, function, or method
        the function used to evaluate the values monitored by the ObjectiveMechanism.  The function can be
        any PsyNeuLink `CombinationFunction` or a Python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ObjectiveMechanism's variable (i.e., its number of
        input_states) and returns a 1d array.

    role : None, LEARNING or CONTROL
        specifies whether the ObjectiveMechanism is used for learning in a `Composition` (in conjunction with a
        `LearningMechanism`), or for control in a Composition (in conjunction with a `ControlMechanism
        <ControlMechanism>`).

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

    name : str
        the name of the ObjectiveMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ObjectiveMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    componentType = OBJECTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ObjectiveCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # class_defaults.variable = None;  Must be specified using either **input_states** or **monitor**
    # kmantel: above needs to be clarified - can class_defaults.variable truly be anything? or should there be some format?
    #   if the latter, we should specify one such valid assignment here, and override _validate_default_variable accordingly
    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <ObjectiveMechanism.function>`

                    :default value: `LinearCombination`
                    :type: `Function`

        """
        function = Parameter(LinearCombination, stateful=False, loggable=False)

    # ObjectiveMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION: LinearCombination,
        })

    standard_output_states = standard_output_states.copy()

    # FIX:  TYPECHECK MONITOR TO LIST OR ZIP OBJECT
    @tc.typecheck
    def __init__(self,
                 monitor=None,
                 default_variable=None,
                 size=None,
                 function=LinearCombination,
                 output_states:tc.optional(tc.any(str, Iterable))=OUTCOME,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # For backward compatibility
        if MONITORED_OUTPUT_STATES in kwargs:
            if monitor:
                raise ObjectiveMechanismError(f'Can\'t specifiy both {repr(MONITOR)} ({monitor}) '
                                              f'and {repr(MONITORED_OUTPUT_STATES)} ({kwargs[MONITORED_OUTPUT_STATES]})'
                                              f' args of {self.name} specified; pick one!')
            warnings.warn(f'Use of {repr(MONITORED_OUTPUT_STATES)} as arg of {self.__class__.__name__} is deprecated;  '
                          f'use {repr(MONITOR)} instead')
            monitor = kwargs.pop(MONITORED_OUTPUT_STATES)
        monitor = monitor or None # deal with possibility of empty list
        input_states = monitor
        if output_states is None or output_states is OUTCOME:
            output_states = [OUTCOME]

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  output_states=output_states,
                                                  function=function,
                                                  params=params)

        self._learning_role = None

        from psyneulink.core.components.states.outputstate import StandardOutputStates
        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         output_states=output_states,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        # This is used to specify whether the ObjectiveMechanism is associated with a ControlMechanism that is
        #    the controller for a Composition;  it is set by the ControlMechanism when it creates the ObjectiveMechanism
        self.for_controller = False

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate **role**, **monitor**, amd **input_states** arguments

        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if ROLE in target_set and target_set[ROLE] and not target_set[ROLE] in {LEARNING, CONTROL}:
            raise ObjectiveMechanismError("\'role\'arg ({}) of {} must be either \'LEARNING\' or \'CONTROL\'".
                                          format(target_set[ROLE], self.name))

    def _instantiate_input_states(self, monitor_specs=None, reference_value=None, context=None):
        """Instantiate InputStates specified in **input_states** argument of constructor or each OutputState
        specified in monitor_specs

        Called during initialization as well as by _add_to_monitor(),
            so must distinguish between initialization and adding to instantiated input_states.

        During initialization, uses **input_states** as specification of InputStates to instantiate;
            if none are specified, instantiates a default InputState

        Otherwise, uses monitor_specs as specification of InputStates to instantiate;
            these will replace any existing InputStates (including a default one)
        """

        # If call is for initialization
        if self.initialization_status == ContextFlags.INITIALIZING:
            # Use self.input_states (containing specs from **input_states** arg of constructor)
            #    or pass off instantiation of default InputState(s) to super
            input_states = self.input_states or None
            input_states = super()._instantiate_input_states(input_states=input_states, context=context)
        else:
            # Parse any spec that is an InputState into the OutputState(s) that project to it
            monitor_specs = _parse_monitor_specs(monitor_specs)

            # Instantiate InputStates corresponding to items specified in monitor
            #     (note: these will replace any existing ones, including the default one created on initialization)
            input_states = super()._instantiate_input_states(input_states=monitor_specs,
                                                             reference_value=reference_value,
                                                             context=context)
        self._name_input_states(input_states)
        return input_states

    def _name_input_states(self, input_states):
        # If InputStates are not already named, name them based on the OutputStates that project to them
        from psyneulink.core.globals.registry import remove_instance_from_registry, register_instance
        if not input_states:
            return
        for state in input_states:
            if not state.path_afferents:
                continue
            if len(state.path_afferents) > 1:
                assert False
            # If the name is not a default name, return
            if not (state.name is InputState.__name__ or InputState.__name__ + '-' in state.name):
                return
            proj = state.path_afferents[0]
            remove_instance_from_registry(registry=self._stateRegistry,
                                          category=INPUT_STATE,
                                          component=state)
            state.name = "Value of {} [{}]".format(proj.sender.owner.name, proj.sender.name)
            register_instance(state, state.name, InputState, self._stateRegistry, INPUT_STATE)

    def add_to_monitor(self, monitor_specs, context=None):
        """Instantiate `OutputStates <OutputState>` to be monitored by the ObjectiveMechanism.

        Used by other Components to add a `State` or list of States to be monitored by the ObjectiveMechanism.
        The **monitor_spec** can be any of the following:
        - MonitoredOutputStateTuple
        - `Mechanism`;
        - `OutputState`;
        - `tuple specification <InputState_Tuple_Specification>`;
        - `State specification dictionary <InputState_Specification_Dictionary>`;
        - list with any of the above.
        If the item is a Mechanism, its `primary OutputState <OutputState_Primary>` is used.
        """
        monitor_specs = list(monitor_specs)

        # If ObjectiveMechanism has only its default InputState and that has no afferent Projections:
        #    delete it and first item of variable
        if len(self.input_states)==1 and self.input_state.name=='InputState-0' and not self.input_state.path_afferents:
            del self.input_states[0]
            self.defaults.variable = []

        # Get reference value
        reference_value = []
        # Get value of each OutputState or, if a Projection from it is specified, then the Projection's value
        for i, spec in enumerate(monitor_specs):
            from psyneulink.core.components.states.inputstate import InputState
            from psyneulink.core.components.system import MonitoredOutputStateTuple
            from psyneulink.core.components.projections.projection import _get_projection_value_shape

            # If it is a MonitoredOutputStateTuple, create InputState specification dictionary
            if isinstance(spec, MonitoredOutputStateTuple):
                # If matrix is specified, let it determine the variable
                if spec.matrix is not None:
                    variable = None
                # Otherwise, use OutputState's value as variable for InputState
                else:
                    variable = spec.output_state.value
                spec = {NAME: spec.output_state.name,
                        VARIABLE: variable,
                        WEIGHT: spec.weight,
                        EXPONENT: spec.exponent,
                        PROJECTIONS: [(spec.output_state, spec.matrix)]}
                monitor_specs[i] = spec

            # Parse spec to get value of OutputState and (possibly) the Projection from it
            input_state = _parse_state_spec(owner=self, state_type = InputState, state_spec=spec)

            # There should be only one ProjectionTuple specified,
            #    that designates the OutputState and (possibly) a Projection from it
            if len(input_state[PARAMS][PROJECTIONS])!=1:
                raise ObjectiveMechanismError("PROGRAM ERROR: Failure to parse item in monitor_specs "
                                              "for {} (item: {})".format(self.name, spec))
            projection_tuple = input_state[PARAMS][PROJECTIONS][0]
            # If Projection is specified, use its value
            if PROJECTION in projection_tuple.projection:
                reference_value.append(projection_tuple.projection[PROJECTION].value)
            # If matrix is specified for Projection, get its receiver dimension
            elif MATRIX in projection_tuple.projection:
                reference_value.append(_get_projection_value_shape(projection_tuple.state,
                                                                   projection_tuple.projection[MATRIX]))
            # Otherwise, use its sender's (OutputState) value
            else:
                reference_value.append(projection_tuple.state.value)

        context.source = ContextFlags.METHOD
        input_states = self._instantiate_input_states(monitor_specs=monitor_specs,
                                                      reference_value=reference_value,
                                                      context=context
                                                      )

        output_states = [[projection.sender for projection in state.path_afferents] for state in input_states]

        self._instantiate_function_weights_and_exponents(context=context)

        return output_states

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

        if hasattr(self.function, WEIGHTS):
            if any(weight is not None for weight in weights):
                self.function.weights = [[weight or DEFAULT_WEIGHT] for weight in weights]
        if hasattr(self.function, EXPONENTS):
            if any(exponent is not None for exponent in exponents):
                self.function.exponents = [[exponent or DEFAULT_EXPONENT] for exponent in exponents]
        assert True

    # # MODIFIED 6/8/19 NEW: [JDC]
    # def _parse_function_variable(self, variable, context=None, context=None):
    #     # CRASHES IN x_or TEST AND LLVM TESTS:
    #     if self.function.variableEncodingDim < self.variableEncodingDim:
    #         return np.squeeze(variable)
    #     # if self.function.variableEncodingDim > self.variableEncodingDim:
    #     #     return np.expand_dims(variable,0)
    #     return variable
    #
    #     # assert False
    # # MODIFIED 6/8/19 END

    @property
    def monitor(self):
        if not isinstance(self.input_states, ContentAddressableList):
            return None
        else:
            monitor = []
            for input_state in self.input_states:
                for projection in input_state.path_afferents:
                    monitor.append(projection.sender)

            return ContentAddressableList(component_type=OutputState,
                                          list=[projection.sender for input_state in self.input_states
                                                for projection in input_state.path_afferents])

    @property
    def monitor_weights_and_exponents(self):
        if hasattr(self.function, WEIGHTS) and self.function.weights is not None:
            weights = self.function.weights
        else:
            weights = [input_state.weight for input_state in self.input_states]
        if hasattr(self.function, EXPONENTS) and self.function.exponents is not None:
            exponents = self.function.exponents
        else:
            exponents = [input_state.exponent for input_state in self.input_states]
        return [(w,e) for w, e in zip(weights,exponents)]

    @monitor_weights_and_exponents.setter
    def monitor_weights_and_exponents(self, weights_and_exponents_tuples):

        weights = [w[0] for w in weights_and_exponents_tuples]
        exponents = [e[1] for e in weights_and_exponents_tuples]
        self._instantiate_weights_and_exponents(weights, exponents)

    # For backward compatibility
    @property
    def monitored_output_states(self):
        return self.monitor

    # For backward compatibility
    @property
    def monitored_output_states_weights_and_exponents(self):
        return self.monitor_weights_and_exponents

    # For backward compatibility
    @monitored_output_states_weights_and_exponents.setter
    def monitored_output_states_weights_and_exponents(self, weights_and_exponents_tuples):
        self.monitor_weights_and_exponents = weights_and_exponents_tuples


def _objective_mechanism_role(mech, role):
    if isinstance(mech, ObjectiveMechanism):
        if mech._role is role:
            return True
        else:
            return False
    else:
        return False


def _parse_monitor_specs(monitor_specs):
    spec_tuple = namedtuple('SpecTuple', 'index spec')
    parsed_specs = []
    specs_to_replace = []
    for i, spec in enumerate(parsed_specs):
        if (isinstance(spec, InputState)
                or (isinstance(spec, dict) and spec[STATE_TYPE] in {INPUT_STATE, InputState})):
            pass # DO PARSING HERE
            if isinstance(spec, InputState):
                pass # Get all projects to it and create InputState specification dictionary with them
            if isinstance(spec, dict) and spec[STATE_TYPE] in {INPUT_STATE, InputState}:
                pass # Get Projections entry and append
            new_spec = None
            specs_to_replace.append(spec_tuple(i, new_spec))

    for spec_tuple in specs_to_replace:
        # # Delete old spec in monitored_specs at specified index
        # del monitor_specs[spec_tuple.index]
        # # Insert new item(s) from spec_tuple into monitor_specs at specified index
        # monitor_specs[spec_tuple.index:spec_tuple.index] = spec_tuple.spec
        monitor_specs[spec_tuple.index] = spec_tuple.spec
    return monitor_specs

# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
#                      ??MAYBE INTEGRATE INTO State MODULE (IN _instantate_state)
# KAM commented out _instantiate_monitoring_projections 9/28/18 to avoid confusion because it never gets called
# @tc.typecheck
# def _instantiate_monitoring_projections(
#     owner,
#     sender_list: tc.any(list, ContentAddressableList),
#     receiver_list: tc.any(list, ContentAddressableList),
#     receiver_projection_specs: tc.optional(list)=None,
#     system=None,
#     context=None
# ):
#
#     from psyneulink.core.components.states.outputstate import OutputState
#     from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
#     from psyneulink.core.components.projections.projection import ProjectionTuple
#
#     receiver_projection_specs = receiver_projection_specs or [DEFAULT_MATRIX] * len(sender_list)
#
#     if len(sender_list) != len(receiver_list):
#         raise ObjectiveMechanismError("PROGRAM ERROR: Number of senders ({}) does not equal number of receivers ({}) "
#                                      "in call to instantiate monitoring projections for {}".
#                                      format(len(sender_list), len(receiver_list), owner.name))
#
#     if len(receiver_projection_specs) != len(receiver_list):
#         raise ObjectiveMechanismError("PROGRAM ERROR: Number of projection specs ({}) "
#                                      "does not equal number of receivers ({}) "
#                                      "in call to instantiate monitoring projections for {}".
#                                      format(len(receiver_projection_specs), len(receiver_list), owner.name))
#
#     # Instantiate InputState with Projection from OutputState specified by sender
#     for sender, receiver, recvr_projs in zip(sender_list, receiver_list, receiver_projection_specs):
#         # IMPLEMENTATION NOTE:  If there is more than one Projection specified for a receiver, only the 1st is used;
#         #                           (there should only be one if a 2-item tuple was used to specify the InputState,
#         #                            however other forms of specifications could produce more)
#         if isinstance(recvr_projs,list) and len(recvr_projs) > 1 and owner.verbosePref:
#             warnings.warn("{} projections were specified for InputState ({}) of {} ;"
#                           "only the first ({}) will be used".
#                           format(len(recvr_projs), receiver.name, owner.name, recvr_projs[0].state.name))
#             projection_spec = recvr_projs[0]
#         else:
#             projection_spec = recvr_projs
#
#         if isinstance(projection_spec, ProjectionTuple):
#             projection_spec = projection_spec.projection
#
#         # IMPLEMENTATION NOTE:  This may not handle situations properly in which the OutputState is specified
#         #                           by a 2-item tuple (i.e., with a Projection specification as its second item)
#         if isinstance(sender, OutputState):
#             # Projection has been specified for receiver and initialization begun, so call deferred_init()
#             if receiver.path_afferents:
#                 if not receiver.path_afferents[0].initialization_status == ContextFlags.DEFERRED_INIT:
#                     raise ObjectiveMechanismError("PROGRAM ERROR: {} of {} already has an afferent projection "
#                                                   "implemented and initialized ({})".
#                                                   format(receiver.name, owner.name, receiver.path_afferents[0].name))
#                 # FIX: 10/3/17 - IS IT OK TO IGNORE projection_spec IF IT IS None?  SHOULD IT HAVE BEEN SPECIFIED??
#                 # FIX:           IN DEVEL, projection_spec HAS BEEN PROPERLY ASSIGNED
#                 if (projection_spec and
#                         not receiver.path_afferents[0].function_params[MATRIX] is projection_spec):
#                     raise ObjectiveMechanismError("PROGRAM ERROR: Projection specification for {} of {} ({}) "
#                                                   "does not match matrix already assigned ({})".
#                                                   format(receiver.name,
#                                                          owner.name,
#                                                          projection_spec,
#                                                          receiver.path_afferents[0].function_params[MATRIX]))
#                 receiver.path_afferents[0].init_args[SENDER] = sender
#                 receiver.path_afferents[0]._deferred_init(context=context)
#             else:
#                 projection_spec = MappingProjection(sender=sender,
#                                                     receiver=receiver,
#                                                     matrix=projection_spec,
#                                                     name=sender.name + ' monitor')
#                 projection_spec._activate_for_compositions(system)
