# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** ContrastiveHebbianMechanism *************************************************

"""
.. _ContrastiveHebbian_Overview:

Overview
--------

A ContrastiveHebbianMechanism is a subclass of `RecurrentTransferMechanism` that is customized for use with the
Contrastive Hebbian learning rule.  See the following references for a description of the learning rule,
its relationship to the backpropagation learning rule, and its use in connectionist networks:

  `Movellan, J. R. (1991). Contrastive Hebbian learning in the continuous Hopfield model. In Connectionist Models
  (pp. 10-17) <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_

  `Xie, X., & Seung, H. S. (2003). Equivalence of backpropagation and contrastive Hebbian learning in a layered network.
  Neural computation, 15(2), 441-454 <https://www.mitpressjournals.org/doi/abs/10.1162/089976603762552988>`_

  `O'reilly, R. C. (2001). Generalization in interactive networks: The benefits of inhibitory competition and Hebbian
  learning. Neural computation, 13(6), 1199-1241 <https://www.mitpressjournals.org/doi/abs/10.1162/08997660152002834>`_

  `Verguts, T., & Notebaert, W. (2008). Hebbian learning of cognitive control: dealing with specific and nonspecific
  adaptation. Psychological review, 115(2), 518 <http://psycnet.apa.org/record/2008-04236-010>`_

The features and operation of a ContrastiveHebbianMechanism that differ from those of a RecurrentTransferMechanism are
described below.

.. _ContrastiveHebbian_Creation:

Creation
--------

When a ContrastiveHebbianMechanism is created, its `has_recurrent_input_state
<RecurrentTransferMechanism.has_recurrent_input_state>` attribute is automatically assigned as `True`, and is
automatically assigned two of its four `Standard OutputStates <ContrastiveHebbianMechanism_Standard_OutputStates>`:
*CURRENT_ACTIVITY_OUTPUT* and *ACTIVITY_DIFFERENT_OUTPUT* (see `below <ContrastiveHebbian_Structure>`). Additional
OutputStates can be specified in the **additional_output_states** argument of its constructor.  A
ContrastiveHebbianMechanism  must have both a `convergence_function <ContrastiveHebbianMechanism.convergence_function>`
and `convergence_criterion <ContrastiveHebbianMechanism.convergence_criterion>`, that determine when `each phase
of execution completes <ContrastiveHebbian_Execution>`.

.. _ContrastiveHebbian_Learning:

A ConstrastiveHebbianMechanism can be configured for learning in the same manner as a
`RecurrentTransferMechanism <Recurrent_Transfer_Learning>`, with the following differences:  it is automatically
assigned `ContrastiveHebbian` as its `learning_function <ContrastiveHebbianMechanism.learning_function>`;
its `learning_condition <RecurrentTransferMechanism.learning_condition>` is automatically assigned as *CONVERGENCE*;
and it is assigned a `MappingProjection` from its *ACTIVITY_DIFFERENCE_OUTPUT* `OutputState
<ContrastiveHebbian_Output>` to the *ACTIVATION_INPUT* of its `learning_mechanism
<ContrastiveHebbianMechanism.learning_mechanism>`.

.. _ContrastiveHebbian_Structure:

Structure
---------

.. _ContrastiveHebbian_Input:

Input
~~~~~

When a ContrastiveHebbianMechanism is created, its `has_recurrent_input_state
<RecurrentTransferMechanism.has_recurrent_input_state>` attribute is automatically assigned as `True`), and thus
it always has at least two `InputStates <InputState>`: *RECURRENT* and *EXTERNAL*.  This is so that, during the
`minus phase of execution <ContrastiveHebbian_Execution>`, it's inpuyt can be restricted to it `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>`.

.. _ContrastiveHebbian_Functions:

Functions
~~~~~~~~~

A ContrastiveHebbianMechanism executes to convergence in each `phase of execution <ContrastiveHebbian_Execution>`
and thus must be assigned both a `convergence_function <ContrastiveHebbianMechanism.convergence_function>` and a
`convergence_criterion <ContrastiveHebbianMechanism.convergence_criterion>`.  The defaults are a `Distance`
Function using the `MAX_DIFF` metric and `absolute_value <Distance.absolute_value>` option, and a
`convergence_criterion <ContrastiveHebbianMechanism.convergence_criterion>` of 0.01.  The `learning_function
<ContrastiveHebbianMechanism.learning_function>` is automatically assigned as `ContrastiveHebbian`,
but it can be replaced by any function that takes two 1d arrays ("activity states") and compares them to determine
the `matrix <MappingProjection.matrix>` of the Mechanism's `recurrent_projection
<ContrastiveHebbianMechanism.recurrent_projection>`.

.. _ContrastiveHebbian_Output:

Output
~~~~~~

A ContrastiveHebbianMechanism is automatically assigned two `OutputStates <OutputState>`: 
*CURRENT_ACTIVITY_OUTPUT* and *ACTIVITY_DIFFERENCE_OUTPUT*.  The former is assigned the value of the `current_activity
<ContrastiveHebbianMechanism.current_activity>` attribute after each `execution of the Mechanism
<ContrastiveHebbian_Execution>`, and the latter the difference between its `plus_phase_activity
<ContrastiveHebbianMechanism.plus_phase_activity>` and `minus_phase_activity
<ContrastiveHebbianMechanism.minus_phase_activity>` at the `completion of execution <ContrastiveHebbian_Execution>`.
If `configured for learning <ContrastiveHebbian_Learning>`, a `MappingProjection` is assigned from the
*ACTIVITY_DIFFERENCE_OUTPUT* `OutputState <ContrastiveHebbian_Output>` to the *ACTIVATION_INPUT* of the
`learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>`.

A ContrastiveHebbianMechanism also has two additional `Standard OutputStates
<ContrastiveHebbianMechanism_Standard_OutputStates>` -- *PLUS_PHASE_ACTIVITY_OUTPUT* and *MINUS_PHASE_OUTPUT* --
that it can be assigned, as well as those of a `RecurrentTransferMechanism
<RecurrentTransferMechanism_Standard_OutputStates>` or `TransferMechanism <TransferMechanism_Standard_OutputStates>`.


.. _ContrastiveHebbian_Execution:

Execution
---------

COMMENT:
    CORRECT/ADD TO DESCRIPTION OF REINTIAILZATION AFTER EACH PHASE
COMMENT

.. _ContrastiveHebbian_Processing:

Processing
~~~~~~~~~~

A ContrastiveHebbianMechanism always executes in two sequential phases, that together constitute a *trial of execution:*

* *plus phase:* in each execution, the inputs received from the *RECURRENT* and *EXTERNAL* `InputStates
  <ContrastiveHebbian_Input>` are combined using the `combination_function
  <ContrastiveHebbianMechanism.combination_function>`, the result of which is passed to `integrator_function
  <ContrastiveHebbianMechanism.integrator_function>` and then `function <ContrastiveHebbianMechanism.function>`.
  The result is assigned to `current_activity <ContrastiveHebbianMechanism.current_activity>`. This is
  compared with the Mechanism's previous `value <ContrastiveHebbianMechanism.value>` using the `convergence_function
  <ContrastiveHebbianMechanism.convergence_function>`, and execution continues until the value returned by that
  function is equal to or below the `convergence_criterion  <ContrastiveHebbianMechanism.convergence_criterion>`
  (i.e., the Mechanism's `is_converged <ContrastiveHebbianMechanism.is_converged>` property is `True`). At that point,
  the *plus phase* is completed, the `value <ContrastiveHebbianMechanism.value>` of the ContrastiveHebbianMechanism
  is assigned to its `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` attribute, and the
  *minus phase* is begun.
..
* *minus phase:* the Mechanism's previous `value <ContrastiveHebbianMechanism.value>` is reinitialized
  to `initial_value <ContrastiveHebbianMechanism.initial_value>`, and then executed using
  only its *RECURRENT* `input <ContrastiveHebbian_Input>`. Otherwise, execution proceeds
  as during the plus phase, completing when `is_converged <ContrastiveHebbianMechanism>` is `True`. At that point,
  the *minus phase* is completed, and the `value <ContrastiveHebbianMechanism.value>` of the Mechanism is assigned
  to `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`.

If `max_passes <ContrastiveHebbianMechanism.max_passes>` is specified, and the number of executions in either phase
reaches that value, an error is generated.  Otherwise, once a trial of execution is complete (i.e, after completion
of the *minus phase*), the following computations and assignments are made:

* the value of `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` is assigned to the
  *CURRENT_ACTIVITY* `OutputState <ContrastiveHebbian_Output>`;
..
* the difference between `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` and
  `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>` is assigned as the `value
  <OutputState.value>` of the the *ACTIVITY_DIFFERENCE_OUTPUT* `OutputState <ContrastiveHebbian_Output>`.

.. _ContrastiveHebbian_Learning_Execution:

Learning
~~~~~~~~

If a ContrastiveHebbianMechanism is `configured for learning <ContrastiveHebbian_Learning>`, at the end of each
`trial of execution <ContrastiveHebbian_Processing>` the `value <OutputState.value>` of its *ACTIVITY_DIFFERENCE_OUTPUT*
`OutputState <ContrastiveHebbian_Output>` is passed to its `learning_mechanism
<ContrastiveHebbianMechanism.learning_mechanism>`.  If the Mechanism is part of a `System`, then the
`learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` is executed during the `execution phase
<System_Execution>` of the System's execution.  Note that this is distinct from the behavior of supervised learning
algorithms (such as `Reinforcement` and `BackPropagation`), that are executed during the `learning phase
<System_Execution>` of a System's execution

.. _ContrastiveHebbian_Class_Reference:

Class Reference
---------------

"""

from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import \
    ContrastiveHebbian, Distance, Function, Linear, LinearCombination, is_function_type, EPSILON
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferMechanism, RECURRENT_INDEX, CONVERGENCE
from psyneulink.globals.keywords import \
    CONTRASTIVE_HEBBIAN_MECHANISM, FUNCTION, HOLLOW_MATRIX, MAX_DIFF, NAME, VARIABLE
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none, parameter_spec

__all__ = [
    'ConstrastiveHebbianError', 'ContrastiveHebbianMechanism', 'CONTRASTIVE_HEBBIAN_OUTPUT',
    'ACTIVITY_DIFFERENCE_OUTPUT', 'CURRENT_ACTIVITY_OUTPUT',
    'MINUS_PHASE_ACTIVITY', 'MINUS_PHASE_OUTPUT', 'PLUS_PHASE_ACTIVITY', 'PLUS_PHASE_OUTPUT',
]

CURRENT_ACTIVITY = 'current_activity'
PLUS_PHASE_ACTIVITY = 'plus_phase_activity'
MINUS_PHASE_ACTIVITY = 'minus_phase_activity'

CURRENT_ACTIVITY_OUTPUT = 'CURRENT_ACTIVITY_OUTPUT'
ACTIVITY_DIFFERENCE_OUTPUT = 'ACTIVITY_DIFFERENCE_OUTPUT'
PLUS_PHASE_OUTPUT = 'PLUS_PHASE_OUTPUT'
MINUS_PHASE_OUTPUT = 'MINUS_PHASE_OUTPUT'

PLUS_PHASE  = True
MINUS_PHASE = False


class ConstrastiveHebbianError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# This is a convenience class that provides list of standard_output_state names in IDE
class CONTRASTIVE_HEBBIAN_OUTPUT():

    """
        .. _ContrastiveHebbianMechanism_Standard_OutputStates:

        `Standard OutputStates <OutputState_Standard>` for `ContrastiveHebbianMechanism` (in addition to those
        for `RecurrentTransferMechanism` and `TransferMechanism`):

        .. _CURRENT_ACTIVITY_OUTPUT:

        *CURRENT_ACTIVITY_OUTPUT* : 1d np.array
            array with current activity of the Mechanism.

        .. _ACTIVITY_DIFFERENCE_OUTPUT:

        *ACTIVITY_DIFFERENCE_OUTPUT* : 1d np.array
            array of element-wise differences in activity between the `plus and minus phases of execution
            <ContrastiveHebbian_Execution>`.

        .. _PLUS_PHASE_OUTPUT:

        *PLUS_PHASE_OUTPUT* : 1d np.array
            array of activity at the end of the `plus phase of execution <ContrastiveHebbian_Execution>`.

        .. _MINUS_PHASE_OUTPUT:

        *MINUS_PHASE_OUTPUT* : 1d np.array
            array of activity at the end of the `minus phase of execution <ContrastiveHebbian_Execution>`

        """
    CURRENT_ACTIVITY_OUTPUT=CURRENT_ACTIVITY_OUTPUT
    ACTIVITY_DIFFERENCE_OUTPUT=ACTIVITY_DIFFERENCE_OUTPUT
    PLUS_PHASE_OUTPUT=PLUS_PHASE_OUTPUT
    MINUS_PHASE_OUTPUT=MINUS_PHASE_OUTPUT


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class ContrastiveHebbianMechanism(RecurrentTransferMechanism):
    """
    ContrastiveHebbianMechanism(                                          \
    default_variable=None,                                                \
    size=None,                                                            \
    function=Linear,                                                      \
    combination_function=LinearCombination,                               \
    matrix=HOLLOW_MATRIX,                                                 \
    auto=None,                                                            \
    hetero=None,                                                          \
    initial_value=None,                                                   \
    noise=0.0,                                                            \
    integration_rate=0.5,                                                 \
    integrator_mode=False,                                                \
    integration_rate=0.5,                                                 \
    clip=[float:min, float:max],                                          \
    convergence_function=Distance(metric=MAX_DIFF, absolute_value=True),  \
    convergence_criterion=0.01,                                           \
    max_passes=None,                                                      \
    enable_learning=False,                                                \
    learning_rate=None,                                                   \
    learning_function=ContrastiveHebbian,                                 \
    additional_output_states=None,                                        \
    params=None,                                                          \
    name=None,                                                            \
    prefs=None)

    Subclass of `RecurrentTransferMechanism` that implements a single-layer auto-recurrent network using two-phases
    of execution and the `Contrastive Hebbian Learning algorithm
    <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_

    COMMENT:
        Description
        -----------
            ContrastiveHebbianMechanism is a Subtype of RecurrentTransferMechanism customized to implement a
            the `ContrastiveHebbian` `LearningFunction`.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <ContrastiveHebbianMechanism.variable>` for
        `function <ContrastiveHebbianMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    combination_function : function : default LinearCombination
        specifies function used to combine the *RECURRENT* and *INTERNAL* `InputStates <Recurrent_Transfer_Structure>`;
        must accept a 2d array with one or two items of the same length, and generate a result that is the same size
        as each of these;  the default function adds the two items.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be any function that takes and returns a 1d array
        of scalar values.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for `recurrent_projection <ConstrastiveHebbianMechanism.recurrent_projection>`;
        see **matrix** argument of `RecurrentTransferMechanism` for details of specification.

    auto : number, 1D array, or None : default None
        specifies matrix with diagonal entries equal to **auto**; see **auto** argument of
        `RecurrentTransferMechanism` for details of specification.

    hetero : number, 2D array, or None : default None
        specifies a hollow matrix with all non-diagonal entries equal to **hetero**;  see **hetero** argument of
        `RecurrentTransferMechanism` for details of specification.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input if `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is `True`.

    noise : float or function : default 0.0
        specifies value added to the result of the `function <ContrastiveHebbianMechanism.function>`, or to the
        result of `integrator_function <ContrastiveHebbianMechanism.integrator_function>` if `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is `True`;  see `noise <ContrastiveHebbianMechanism.noise>`
        for additional details.

    integration_rate : float : default 0.5
        the rate used for exponential time averaging of input when `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is `True`; see `integration_rate
        <TransferMechanism.integration_rate>` for additional details.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>`;
        see `clip <TransferMechanism.clip>` for additional details.

    convergence_function : function : default Distance(metric=MAX_DIFF, absolute_value=True)
        specifies the function that determines when `each phase of execution completes<ContrastiveHebbian_Execution>`,
        by comparing `current_activity <ContrastiveHebbianMechanism.current_activity>` with the previous `value
        <ContrastiveHebbian.value>` of the Mechanism;  can be any function that takes two 1d arrays of the same length
        as `variable <ContrastiveHebbianMechanism.variable>` and returns a scalar value. The default is the `Distance`
        Function, using the `MAX_DIFF` metric and `absolute_value <Distance.absolute_value>` option, which computes
        the elementwise difference between two arrays and returns the difference with the maximum absolute value.

    convergence_criterion : float : default 0.01
        specifies the value of the `convergence_function <ContrastiveHebbianMechanism.convergence_function>`
        used to determine when `each phase of execution completes <ContrastiveHebbian_Execution>`.

    max_passes : int : default 1000
        specifies maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the `convergence_criterion
        <ContrastiveHebbianMechanism.convergence_criterion>`, after which an error occurs; if `None` is specified,
        execution may continue indefinitely or until an interpreter exception is generated.

    enable_learning : boolean : default False
        specifies whether the Mechanism should be `configured for learning <ConstrastiveHebbian_Learning>`.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <ContrastiveHebbianMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <ContrastiveHebbianMechanism.learning_rate>` for details).

    learning_function : function : default ContrastiveHebbian
        specifies the function for the LearningMechanism if `learning is specified <ContrastiveHebbian_Learning>` for
        the ContrastiveHebbianMechanism.  It can be any function so long as it takes a list or 1d array of numeric
        values as its `variable <Function_Base.variable>` and returns a sqaure matrix of numeric values with the same
        dimensions as the length of the input.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <ContrastiveHebbianMechanism.name>`
        specifies the name of the ContrastiveHebbianMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ContrastiveHebbianMechanism; see `prefs <ContrastiveHebbianMechanism.prefs>`
        for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <ContrastiveHebbianMechanism.variable>`.

    combination_function : function
        used to combine `value <InputState.value>` of the *RECURRENT* and *EXTERNAL* `inputs
        <ContrastiveHebbian_Input>`.  The default function adds them.

    function : Function
        used to transform the input and generate the Mechanism's `value <ContrastiveHebbianMechanism.value>`.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        to its `primary InputState <Mechanism_InputStates>`;

    initial_value :  value, list or np.ndarray
        determines the starting value for time-averaged input if `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is `True`. See TransferMechanism's `initial_value
        <TransferMechanism.initial_value>` for additional details.

    noise : float or function
        When `integrator_mode <ContrastiveHebbianMechanism.integrator_mode>` is set to `True`, noise is passed into the
        `integrator_function <ContrastiveHebbianMechanism.integrator_function>`. Otherwise, noise is added to the result
        of the `function <ContrastiveHebbianMechanism.function>`.  See TransferMechanism's `noise
        <TransferMechanism.noise>` for additional details.

    integrator_mode:
        determines whether input is first processed by `integrator_function
        <ContrastiveHebbianMechanism.integrator_function>` before being passed to `function
        <ContrastiveHebbianMechanism.function>`; see TransferMechanism's `integrator_mode
        <TransferMechanism.integrator_mode>` for additional details.

    integrator_function:
        used by the Mechanism when it executes if `integrator_mode <ContrastiveHebbianMechanism.integrator_mode>` is
        `True`.  Uses the `integration_rate  <ContrastiveHebbianMechanism.integration_rate>` parameter
        of the ContrastiveHebbianMechanism as the `rate <Integrator.rate>` of the ContrastiveHebbianMechanism's
        `integrator_function`; see TransferMechanism's `integrator_function <TransferMechanism.integrator_function>`
        for additional details.

    integration_rate : float
        the rate used for exponential time averaging of input when `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is set to `True`;  see TransferMechanism's
        `integration_rate <TransferMechanism.integration_rate>` for additional details.

    clip : list [float, float]
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>`
        see TransferMechanism's `clip <TransferMechanism.clip>` for additional details.

    current_activity : 1d array of floats
        the value of the actvity of the ContrastiveHebbianMechanism at `the current step of execution
        <ContrastiveHebbian_Execution>`.

    plus_phase_activity : 1d array of floats
        the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>` at the end of the
        `plus phase of execution <ContrastiveHebbian_Execution>`.

    minus_phase_activity : 1d array of floats
        the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>` at the end of the
        `minus phase of execution <ContrastiveHebbian_Execution>`.

    previous_value : 1d array of floats
        the value of `current_activity <ContrastiveHebbianMechanism.current_activity>` on the `previous
        execution in the current phase <ContrastiveHebbian_Execution>`.

    is_converged : bool
        `True` when the value returned by `converge_function <ContrastiveHebbianMechanism.convergence_function>`.
        is less than or equal to the `converge_criterion <ContrastiveHebbianMechanism.convergence_criterion>`;
        used by the ContrastiveHebbianMechanism to determine when `each phase of execution is complete
        <ContrastiveHebbian_Execution>`.

    convergence_function : function
        compares the value of `current_activity <ContrastiveHebbianMechanism.current_activity>` with the previous
        `value <ContrastiveHebbianMechanism.value>` of the Mechanism and returns a scalar value; used to determine
        when `each phase of execution is complete <ContrastiveHebbian_Execution>` (i.e., when `is_converged
        <ContrastiveHebbianMechanism.is_converged>` is `True`.
    
    convergence_criterion : float
        determines the value of `convergence_function <ContrastiveHebbianMechanism.convergence_function>` at which
        `each phase of execution completes <ContrastiveHebbian_Execution>`.

    max_passes : int or None
        determines the maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the `convergence_criterion
        <RecurrentTransferMechanism.convergence_criterion>`, after which an error occurs;
        if `None` is specified, execution may continue indefinitely or until an interpreter exception is generated.

    learning_enabled : bool
        indicates whether `learning is configured <ContrastiveHebbian_Learning>`;  see `learning_enabled
        <RecurrentTransferMechanism.learning_enabled>` of RecurrentTransferMechanism for additional details.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <ContrastiveHebbian_Learning>`, and used to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values
        specifies the learning rate used by the `learning_function <ContrastiveHebbianMechanism.learning_function>`
        of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignment).

    learning_function : function
        the function used by the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>` if `learning is configured
        <ContrastiveHebbian_Learning>`.

    value : 2d np.array
        result of executing `function <ContrastiveHebbianMechanism.function>`; same value as first item of
        `output_values <ContrastiveHebbianMechanism.output_values>`.

    output_states : Dict[str: OutputState]
        an OrderedDict with the following `OutputStates <OutputState>` by default:

        * *CURRENT_ACTIVITY_OUTPUT* -- the  `primary OutputState  <OutputState.primary>` of the Mechanism, the
          `value <OutputState.value>` of which is a 1d array containing the activity of the ContrastiveHebbianMechanism
          after each execution;  at the end of an execution sequence (i.e., when `is_finished
          <ContrastiveHebbianMechanism.is_finished>` is `True`), it is assigned the value of `plus_phase_activity
          <ContrastiveHebbianMechanism.plus_phase_activity>`.

        * *ACTIVITY_DIFFERENCE_OUTPUT*, the `value <OutputState.value>` of which is a 1d array with the element-wise
          differences in activity between the plus and minus phases at the end of an execution sequence.

    output_values : List[1d np.array]
        a list with the following items by default:
        * **current_activity_output** at the end of an execution.
        * **activity_difference_output** at the end of an execution.

    name : str
        the name of the ContrastiveHebbianMechanism; if it is not specified in the **name** argument of the constructor,
        a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ContrastiveHebbianMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of ContrastiveHebbianMechanism : ContrastiveHebbianMechanism

    """
    componentType = CONTRASTIVE_HEBBIAN_MECHANISM

    class ClassDefaults(RecurrentTransferMechanism.ClassDefaults):
        variable = np.array([[0]])

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:CURRENT_ACTIVITY_OUTPUT,
                                    VARIABLE:CURRENT_ACTIVITY},
                                   {NAME:ACTIVITY_DIFFERENCE_OUTPUT,
                                    VARIABLE:[PLUS_PHASE_ACTIVITY, MINUS_PHASE_ACTIVITY],
                                    FUNCTION: lambda v: v[0] - v[1]},
                                   {NAME:PLUS_PHASE_OUTPUT,
                                    VARIABLE:PLUS_PHASE_ACTIVITY},
                                   {NAME:MINUS_PHASE_OUTPUT,
                                    VARIABLE:MINUS_PHASE_ACTIVITY},
                                   ])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict)) = None,
                 combination_function:is_function_type=LinearCombination,
                 function=Linear,
                 matrix=HOLLOW_MATRIX,
                 auto=None,
                 hetero=None,
                 initial_value=None,
                 noise=0.0,
                 integration_rate: is_numeric_or_none=0.5,
                 integrator_mode:bool=False,
                 clip=None,
                 convergence_function:tc.any(is_function_type)=Distance(metric=MAX_DIFF, absolute_value=True),
                 convergence_criterion:float=0.01,
                 max_passes:tc.optional(int)=1000,
                 enable_learning:bool=False,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = ContrastiveHebbian,
                 additional_output_states:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None):

        """Instantiate ContrastiveHebbianMechanism"""

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        output_states = [CURRENT_ACTIVITY_OUTPUT, ACTIVITY_DIFFERENCE_OUTPUT]
        if additional_output_states:
            if isinstance(additional_output_states, list):
                output_states += additional_output_states
            else:
                output_states.append(additional_output_states)

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(output_states=output_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         combination_function=combination_function,
                         function=function,
                         matrix=matrix,
                         auto=auto,
                         hetero=hetero,
                         has_recurrent_input_state=True,
                         initial_value=initial_value,
                         noise=noise,
                         integrator_mode=integrator_mode,
                         integration_rate=integration_rate,
                         clip=clip,
                         convergence_function=convergence_function,
                         convergence_criterion=convergence_criterion,
                         max_passes=max_passes,
                         enable_learning=enable_learning,
                         learning_rate=learning_rate,
                         learning_function=learning_function,
                         learning_condition=CONVERGENCE,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs)

    def _instantiate_attributes_after_function(self, context=None):

        # Assign these after instantiation of function, since they are initialized in _execute (see below)
        self.attributes_dict_entries.update({CURRENT_ACTIVITY:CURRENT_ACTIVITY,
                                             PLUS_PHASE_ACTIVITY:PLUS_PHASE_ACTIVITY,
                                             MINUS_PHASE_ACTIVITY:MINUS_PHASE_ACTIVITY})

        super()._instantiate_attributes_after_function(context=context)


    def _execute(self,
                 variable=None,
                 function_variable=None,
                 runtime_params=None,
                 context=None):

        if self.context.initialization_status == ContextFlags.INITIALIZING:
            # Set plus_phase, minus_phase activity, current_activity and initial_value
            #    all  to zeros with size of Mechanism's array
            self._initial_value = self.current_activity = self.plus_phase_activity = self.minus_phase_activity = \
                self.input_state.socket_template
            self.execution_phase = None

        # Initialize execution_phase
        if self.execution_phase is None:
            self.execution_phase = PLUS_PHASE
        # # USED FOR TEST PRINT BELOW:
        # curr_phase = self.execution_phase

        if self.is_finished == True:
            # If current execution follows completion of a previous trial,
            #    zero activity for input from recurrent projection so that
            #    input does not contain residual activity of previous trial
            variable[RECURRENT_INDEX] = self.input_state.socket_template

        self.is_finished = False

        # Note _parse_function_variable selects actual input to function based on execution_phase
        current_activity = super()._execute(variable,
                                            runtime_params=runtime_params,
                                            context=context)

        try:
            # self.current_activity = np.squeeze(current_activity)
            current_activity = np.squeeze(current_activity)
            # Set value of primary OutputState to current activity
            self.current_activity = current_activity
        except:
            assert False

        if self._previous_mech_value is None:
            return current_activity

        if self.is_converged:
            # Terminate if this is the end of the minus phase
            if self.execution_phase == MINUS_PHASE:
                # Store activity from last execution in minus phase
                self.minus_phase_activity = current_activity
                # Set value of primary outputState to activity at end of plus phase
                self.current_activity = self.plus_phase_activity
                self.is_finished = True

            # Otherwise, prepare for start of minus phase on next execution
            else:
                # Store activity from last execution in plus phase
                self.plus_phase_activity = current_activity
                # self.plus_phase_activity = self.current_activity
                # Use initial_value attribute to initialize, for the minus phase,
                #    both the integrator_function's previous_value
                #    and the Mechanism's current activity (which is returned as it input)
                self.reinitialize(self.initial_value)
                self.current_activity = self.initial_value

            # # USED FOR TEST PRINT BELOW:
            # curr_phase = self.execution_phase

            # Switch execution_phase
            self.execution_phase = not self.execution_phase
        # MODIFIED 7/7/18 END

        # # TEST PRINT:
        # print("--------------------------------------------",
        #       "\nTRIAL: {}  PASS: {}".format(self.current_execution_time.trial, self.current_execution_time.pass_),
        #       '\nphase: ', 'PLUS' if curr_phase == PLUS_PHASE else 'MINUS',
        #       '\nvariable: ', variable,
        #       '\ninput:', self.function_object.variable,
        #       '\nMATRIX:', self.matrix,
        #       '\ncurrent activity: ', self.current_activity,
        #       '\ndiff: ', self._output,
        #       '\nis_finished: ', self.is_finished
        #       )

        return current_activity
        # return self.current_activity

    def _parse_function_variable(self, variable, context=None):

        try:
            if self.execution_phase == PLUS_PHASE:
                # Combine RECURRENT and EXTERNAL inputs
                variable = self.combination_function.execute(variable)
            else:
                # Only use RECURRENT input
                variable = variable[RECURRENT_INDEX]                     # Original
                # variable = np.zeros_like(variable[RECURRENT_INDEX])        # New

        except:
            variable = variable[RECURRENT_INDEX]

        return super(RecurrentTransferMechanism, self)._parse_function_variable(variable, context)

    @property
    def _learning_signal_source(self):
        '''Override default to use ACTIVITY_DIFFERENCE_OUTPUT as source of learning signal
        '''
        return self.output_states[ACTIVITY_DIFFERENCE_OUTPUT]