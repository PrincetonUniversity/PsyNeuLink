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

*Size*
~~~~~~

The **input_size** argument of the constructor must always be specified (this is comparable to specifying the
**size** or *default_variable** arugments of other types of `Mechanism`).  If it is specified on its own,
it determines the total number of processing units.  If either the **hidden_size** and/or **target_size** arguments
are specified, then those units are treated as distinct from the input units (see `ContrastiveHebbian_Execution` for
details).

*Phases*
~~~~~~~~

A ContrastiveHebbianMechanism `executes in two phases <ContrastiveHebbian_Execution>`, and has
**minus_phase_termination_condition** and **plus_phase_termination_condition** arguments, and corresponding
**minus_phase_termination_criterion** and **plus_phase_termination_criterion** arguments, that determine when the
respective phases of execution are terminated.  Other parameters can also be configured that influence processing (see
`ContrastiveHebbian_Execution`).

*OututStates*
~~~~~~~~~~~~~

The Mechanism is automatically assigned three of its five `Standard OutputStates
<ContrastiveHebbianMechanism_Standard_OutputStates>`: *OUTPUT_ACTIVITY_OUTPUT*, *CURRENT_ACTIVITY_OUTPUT*,
and *ACTIVITY_DIFFERENT_OUTPUT* (see `below <ContrastiveHebbian_Output>`). Additional OutputStates can be specified
in the **additional_output_states** argument of the constructor.

.. _ContrastiveHebbian_Learning:

*Learning*
~~~~~~~~~~

A ContrastiveHebbianMechanism can be configured for learning either by specifying **enable_learning** as `True` or using
the `configure_learning <RecurrentTransferMechanism.configure_learning>` method, with the following differences from a
standard `RecurrentTransferMechanism <Recurrent_Transfer_Learning>`:  it is automatically assigned `ContrastiveHebbian`
as its `learning_function <ContrastiveHebbianMechanism.learning_function>`; its `learning_condition
<RecurrentTransferMechanism.learning_condition>` is automatically assigned as *CONVERGENCE*; and it is assigned a
`MappingProjection` from its *ACTIVITY_DIFFERENCE_OUTPUT* (rather than its `primary <OutputState_Primary>`)
`OutputState <ContrastiveHebbian_Output>` to the *ACTIVATION_INPUT* of its `learning_mechanism
<ContrastiveHebbianMechanism.learning_mechanism>`.

.. _ContrastiveHebbian_SIMPLE_HEBBIAN:

*SIMPLE_HEBBIAN mode*
~~~~~~~~~~~~~~~~~~~~~

This replicates the function of a standard RecurrentTransferMechanism using Hebbian Learning (e.g., for validation or
ease in comparing processing outcomes).  It is configured by specifying the **mode** argument as *SIMPLE_HEBBIAN*,
which automatically makes the following assignments:

|    **separated** = `False`;
|    **clamp** = `SOFT_CLAMP`;
|    **learning_function** = `Hebbian`.
|
These assignments override any others made in the constructor, and the **hidden_size** and **target_size** arguments
are ignored.


.. _ContrastiveHebbian_Structure:

Structure
---------

.. _ContrastiveHebbian_Input:

*Input*
~~~~~~~

A ContrastiveHebbianMechanism always has two, and possibly three `InputStates <InputState>`: 

    * *INPUT:* receives external input to the Mechanism;
    ..
    * *RECURRENT:* receives the `value <MappingProjection.value>` of the Mechanism's `recurrent_projection
      <ContrastiveHebbianMechanism.recurrent_projection>`;
    ..
    * *TARGET:* only implemented if **target_size** is specified, **separated = `True` (default), and
      mode is not `SIMPLE_HEBBIAN <ContrastiveHebbian_SIMPLE_HEBBIAN>`;  receives the `target <Run.target>`
      specified in the `run <System.run>` method of any `System` to which the Mechanism belongs.

The sizes of these are determined by arguments in its constructor, which generally conform to one of two
configurations.

|    **Standard Configuration** --   if **input_size**, **hidden_size** and **target_size** are all specified,
|    the sizes of the InputStates are as follows:
|
|        *INPUT*:  size = **input_size**
|        *RECURRENT*:  size = **input_size** + **hidden_size** + **target_size**
|        *TARGET*:  size = **input_size**
|
|    **Simple Configuration** -- if **target_size** = `None` or 0,  **separated** = `False`, and/or **mode** =
|    *SIMPLE_HEBBIAN*, the sizes of the InputStates are as follows:
|
|        *INPUT*:  size = **input_size**
|        *RECURRENT*:  size = **input_size** + **hidden_size**
|        *TARGET*:  Not implemented
|
.. note::
   If **separated** = `False` and **target_size** is specified, a *TARGET* InputState will be created, but
   **target_size** must equal **input_size** or an error will be generated.

The values of **input_size**, **hidden_size** and **target_size** are assigned to the Mechanism's
`input_size <ContrastiveHebbianMechanism.input_size>`, `hidden_size <ContrastiveHebbianMechanism.hidden_size>`,
and `target_size <ContrastiveHebbianMechanism.target_size>` attributes, respectively.

.. _ContrastiveHebbian_Fields:

The `input_size <ContrastiveHebbianMechanism.input_size>` and `target_size <ContrastiveHebbianMechanism.target_size>`
(if **separated** is `True`) attribute(s), together with the `separated <ContrastiveHebbianMechanism>` attribute
are used to define the fields of the *RECURRENT* InputState's `value <InputState.value>` and the `current_activity
<ContrastiveHebbianMechanism>` attribute used for updating values during `execution <ContrastiveHebbian_Processing>`,
as follows:

    * *input_field:*  the leftmost number of elements determined by `input_size
      <ContrastiveHebbianMechanism.input_size>`;
    ..
    * *hidden_field:*  the elements of the *RECURRENT* InputState and `current_activity <ContrastiveHebbianMechanism>`
      that are not within the *input_field* and/or *target_field*;
    ..
    * *target_field:* the rightmost number of elements determined by `target_size
      <ContrastiveHebbianMechanism.target_size>` if `separated <ContrastiveHebbianMechanism.separated>` is `True`;
      otherwise, the same as *input_field*.

.. _ContrastiveHebbian_Functions:

*Functions*
~~~~~~~~~~~

In addition to its primary `function <ContrastiveHebbianMechanism.function>`, if either the
`minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>` or
`plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`
is specified as *CONVERGENCE*, then its `convergence_function <ContrastiveHebbianMechanism.convergence_function>`,
that is used to determine when the corresponding `phase of execution <ContrastiveHebbian_Execution>` is complete.
Its `learning_function <ContrastiveHebbianMechanism.learning_function>` is automatically assigned as
`ContrastiveHebbian`, but this can be replaced by any function that takes two 1d arrays ("activity states") and
compares them to determine the `matrix <MappingProjection.matrix>`  of the Mechanism's `recurrent_projection
<ContrastiveHebbianMechanism.recurrent_projection>`.  If **mode** is specified as `SIMPLE_HEBBIAN
<ContrastiveHebbian_SIMPLE_HEBBIAN>`), the default `function <ContrastiveHebbianMechanism.function>` is `Hebbian`,
but can be replaced by any function that takes and returns a 1d array.

.. _ContrastiveHebbian_Output:

*Output*
~~~~~~~~

A ContrastiveHebbianMechanism is automatically assigned three `OutputStates <OutputState>`: 

* *OUTPUT_ACTIVITY_OUTPUT:* assigned as the `primary OutputState <OutputState_Primary>`, and contains the pattern
  of activity the Mechanism is trained to generate.  If **target_size** is specified, then it has the same size as
  the *TARGET* InputState;  if **target_size** is not specified or is 0, if `separated
  <ContrastiveHebbianMechanism.separated>` is `False`, or if mode is *SIMPLE_HEBBIAN*, then the size of the
  *OUTPUT_ACTIVITY_OUTPUT* OutputState is the same as the *INPUT* InputState (see `ContrastiveHebbian_Input`).
..
* *CURRENT_ACTIVITY_OUTPUT:* assigned the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>`
  attribute after each `execution <ContrastiveHebbian_Execution>` of the Mechanism, which contains the activity of all
  processing units in the Mechanism (input, hidden and/or target);  has the same size as the *RECURRENT* InputState.
..
* *ACTIVITY_DIFFERENCE_OUTPUT:* assigned the difference between `plus_phase_activity
  <ContrastiveHebbianMechanism.plus_phase_activity>` and `minus_phase_activity
  <ContrastiveHebbianMechanism.minus_phase_activity>` at the `completion of execution
  <ContrastiveHebbian_Execution>`. If `configured for learning <ContrastiveHebbian_Learning>`, a `MappingProjection`
  is assigned from the *ACTIVITY_DIFFERENCE_OUTPUT* `OutputState <ContrastiveHebbian_Output>` to the
  *ACTIVATION_INPUT* of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>`.

A ContrastiveHebbianMechanism also has two additional `Standard OutputStates
<ContrastiveHebbianMechanism_Standard_OutputStates>` -- *MINUS_PHASE_OUTPUT* and *PLUS_PHASE_ACTIVITY_OUTPUT* --
that it can be assigned, as well as those of a `RecurrentTransferMechanism
<RecurrentTransferMechanism_Standard_OutputStates>` or `TransferMechanism <TransferMechanism_Standard_OutputStates>`.


.. _ContrastiveHebbian_Execution:

Execution
---------

.. _ContrastiveHebbian_Processing:

*Processing*
~~~~~~~~~~~~

A ContrastiveHebbianMechanism always executes in two sequential phases, that together constitute a *trial of execution:*

.. _ContrastiveHebbian_Minus_Phase:

* *minus phase:* in each execution, the *RECURRENT* InputState's `value <InputState.value>` (received from the
  `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`) is combined with the *INPUT*
  InputState's `value <InputState.value>`. The result of `function <ContrastiveHebbianMechanism.function>` is
  assigned to `current_activity <ContrastiveHebbianMechanism.current_activity>`.  The Mechanism is executed repeatedly
  until its `minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>` is met.
  At that point, the *minus phase* is completed, the `value <ContrastiveHebbianMechanism.value>` of the
  ContrastiveHebbianMechanism is assigned to its `minus_phase_activity
  <ContrastiveHebbianMechanism.minus_phase_activity>` attribute, and the *plus phase* is begun.
..

.. _ContrastiveHebbian_Plus_Phase:

* *plus phase:*  if `continuous <ContrastiveHebbianMechanism.continuous>` is `False`, then `current_activity
  <ContrastiveHebbianMechanism.current_activity>` and the Mechanism's `previous_value
  <ContrastiveHebbianMechanism.previous_value>` attribute are reinitialized to `initial_value
  <ContrastiveHebbianMechanism.initial_value>`;  otherwise, these retain their value from the last execution in the
  *minus phase*.  In either case, the *RECURRENT* InputState's `value <InputState.value>` is combined with the *INPUT*
  InputState's `value <InputState.value>` (as during the `minus_phase
  <ContrastiveHebbianMechanism.minus_phase_activity>`) as well as that of *TARGET* InputState (if that is `specified
  <ContrastiveHebbian_Input>`). If `separated <ContrastiveHebbianMechanism.separated>` is `True` (which it is by
  default), then the `value <InputState.value>` of the *INPUT* `InputState <ContrastiveHebbian_Input>` is combined with
  the `input_field <ContrastiveHebbian_Fields>` of the *RECURRENT* InputState's `value <InputState.value>`,
  and *TARGET* is combined with the `target_field <ContrastiveHebbian_Fields>`; otherwise, both are combined with the
  `input_field <ContrastiveHebbian_Fields>`.  If `hidden_size <ContrastiveHebbianMechanism.hidden_size>` is
  specified, then the `hidden_field <ContrastiveHebbian_Fields>` of the *RECURRENT* InputState's  `value
  <InputState.value>` is determined only by the `value <AutoAssociativeProjection.value>` of the Mechanism's
  `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.  Execution then proceeds as during the
  *minus phase*, completing when its `plus_phase_termination_condition
  <ContrastiveHebbianMechanism.plus_phase_termination_condition>` is met.  At that point, the *plus phase* is
  completed, and the `value <ContrastiveHebbianMechanism.value>` of the Mechanism is assigned to
  `plus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`.

The `value <InputState.value>` of the *INPUT*, and possibly *TARGET*, InptutState(s) are combined with that of its
*RECURRENT* InputState using the `combination_function <ContrastiveHebbianMechanism.combination_function>`. The
manner in which these are combined is determined by the `clamp <ContrastiveHebbianMechanism.clamp>` attribute: if
it is *HARD_CLAMP* they are used to replace the corresponding fields of *RECURRENT*;  if it is *SOFT_CLAMP*, *INPUT*
(and possibly *TARGET*) are added to *RECURRENT*; .  The result is passed to the Mechanism's `integrator_function
<ContrastiveHebbianMechanism.integrator_function>` (if `integrator_mode <ContrastiveHebbianMechanism.integrator_mode>`
is `True`) and then its `function <ContrastiveHebbianMechanism.function>`.

If the termination condition for either phase is specified as *CONVERGENCE*, it uses the Mechanism's
`convergence_function <ContrastiveHebbianMechanism.convergence_function>`, together with the convergence criterion
for that phase, to determine when execution of that phase terminates.  If
`max_passes <ContrastiveHebbianMechanism.max_passes>` is specified, and the number of executions in either phase
reaches that value, an error is generated.  Otherwise, once a trial of execution is complete (i.e, after completion
of the *minus phase*), the following computations and assignments are made:

* if a *TARGET* InputState `has been specified <ContrastiveHebbian_Input>`, then the `target field
  <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>` is assigned as
  `value <OutputState.value>` of *OUTPUT_ACTIVITY_OUTPUT* `OutputState <ContrastiveHebbian_Output>`;  otherwise,
  it is assigned the value of the `input_field <ContrastiveHebbian_Fields>` of `current_activity
  <ContrastiveHebbianMechanism.current_activity>`.
..
* `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` is assigned as the `value
  <OutputState.value>` of the *CURRENT_ACTIVITY* `OutputState <ContrastiveHebbian_Output>`;
..
* the difference between `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` and
  `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>` is assigned as the `value
  <OutputState.value>` of the *ACTIVITY_DIFFERENCE_OUTPUT* `OutputState <ContrastiveHebbian_Output>`.

.. _ContrastiveHebbian_Learning_Execution:

*Learning*
~~~~~~~~~~

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

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.learningfunctions import ContrastiveHebbian, Hebbian
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear, get_matrix
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONTRASTIVE_HEBBIAN_MECHANISM, COUNT, FUNCTION, HARD_CLAMP, HOLLOW_MATRIX, MAX_ABS_DIFF, NAME, SIZE, SOFT_CLAMP, TARGET, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import CONVERGENCE, RECURRENT, RECURRENT_INDEX, RecurrentTransferMechanism

__all__ = [
    'ContrastiveHebbianError', 'ContrastiveHebbianMechanism', 'CONTRASTIVE_HEBBIAN_OUTPUT',
    'ACTIVITY_DIFFERENCE_OUTPUT', 'CURRENT_ACTIVITY_OUTPUT', 'SIMPLE_HEBBIAN', 'INPUT',
    'MINUS_PHASE_ACTIVITY', 'MINUS_PHASE_OUTPUT', 'PLUS_PHASE_ACTIVITY', 'PLUS_PHASE_OUTPUT'
]

INPUT = 'INPUT'

INPUT_SIZE = 'input_size'
HIDDEN_SIZE = 'hidden_size'
TARGET_SIZE = 'target_size'
SEPARATED = 'separated'

SIMPLE_HEBBIAN = 'SIMPLE_HEBBIAN'

INPUT_INDEX = 0
RECURRENT_INDEX = 1
TARGET_INDEX = 2

OUTPUT_ACTIVITY = 'output_activity'
CURRENT_ACTIVITY = 'current_activity'
MINUS_PHASE_ACTIVITY = 'minus_phase_activity'
PLUS_PHASE_ACTIVITY = 'plus_phase_activity'

OUTPUT_ACTIVITY_OUTPUT = 'OUTPUT_ACTIVITY_OUTPUT'
CURRENT_ACTIVITY_OUTPUT = 'CURRENT_ACTIVITY_OUTPUT'
ACTIVITY_DIFFERENCE_OUTPUT = 'ACTIVITY_DIFFERENCE_OUTPUT'
MINUS_PHASE_OUTPUT = 'MINUS_PHASE_OUTPUT'
PLUS_PHASE_OUTPUT = 'PLUS_PHASE_OUTPUT'

MINUS_PHASE = False
PLUS_PHASE  = True


class ContrastiveHebbianError(Exception):
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

        .. _OUTPUT_ACTIVITY_OUTPUT:

        *OUTPUT_ACTIVITY_OUTPUT* : 1d np.array
            array with activity of the `target_field <ContrastiveHebbian_Fields>` of `current_activity
            <ContrastiveHebbianMechanism.current_activity>` if a *TARGET* `InputState is specified
            <ContrastiveHebbian_Input>`;  otherwise, has activity of the `input_field <ContrastiveHebbian_Fields>` of
            `current_activity <ContrastiveHebbianMechanism.current_activity>`.

        .. _CURRENT_ACTIVITY_OUTPUT:

        *CURRENT_ACTIVITY_OUTPUT* : 1d np.array
            array with `current_activity <ContrastiveHebbianMechanism.current_activity>`.

        .. _ACTIVITY_DIFFERENCE_OUTPUT:

        *ACTIVITY_DIFFERENCE_OUTPUT* : 1d np.array
            array of element-wise differences between `plus_phase_activity
            <ContrastiveHebbianMechanism.plus_phase_activity>` and `minus_phase_activity
            <ContrastiveHebbianMechanism.minus_phase_activity>`.

        .. _MINUS_PHASE_OUTPUT:

        *MINUS_PHASE_OUTPUT* : 1d np.array
            array `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`
            (i.e., activity at the end of the `minus phase of execution <ContrastiveHebbian_Minus_Phase>`.

        .. _PLUS_PHASE_OUTPUT:

        *PLUS_PHASE_OUTPUT* : 1d np.array
            array `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>`
            (i.e., activity at the end of the `plus phase of execution <ContrastiveHebbian_Plus_Phase>`.


        """
    CURRENT_ACTIVITY_OUTPUT=CURRENT_ACTIVITY_OUTPUT
    ACTIVITY_DIFFERENCE_OUTPUT=ACTIVITY_DIFFERENCE_OUTPUT
    MINUS_PHASE_OUTPUT=MINUS_PHASE_OUTPUT
    PLUS_PHASE_OUTPUT=PLUS_PHASE_OUTPUT


def _CHM_output_activity_getter(owning_component=None, context=None):
    return owning_component.parameters.current_activity._get(context)[owning_component.target_start:owning_component.target_end]


def _CHM_input_activity_getter(owning_component=None, context=None):
    return owning_component.parameters.current_activity._get(context)[:owning_component.input_size]


def _CHM_hidden_activity_getter(owning_component=None, context=None):
    if owning_component.hidden_size:
        return owning_component.parameters.current_activity._get(context)[owning_component.input_size:owning_component.target_start]


def _CHM_target_activity_getter(owning_component=None, context=None):
    if owning_component.target_size:
        return owning_component.parameters.current_activity._get(context)[owning_component.target_start:owning_component.target_end]


class ContrastiveHebbianMechanism(RecurrentTransferMechanism):
    """
    ContrastiveHebbianMechanism(                                    \
                input_size=2,                                       \
                hidden_size=None,                                   \
                target_size=None,                                   \
                separated:bool=True,                                \
                mode=None,                                          \
                continuous=True,                                    \
                clamp=HARD_CLAMP,                                   \
                function=Linear,                                    \
                integrator_functon=AdapativeIntegrator,             \
                combination_function=LinearCombination,             \
                matrix=HOLLOW_MATRIX,                               \
                auto=None,                                          \
                hetero=None,                                        \
                initial_value=None,                                 \
                noise=0.0,                                          \
                integration_rate=0.5,                               \
                integrator_mode=False,                              \
                integration_rate=0.5,                               \
                clip=[float:min, float:max],                        \
                minus_phase_termination_condition = CONVERGENCE,    \
                minus_phase_termination_criterion=.01,              \
                plus_phase_termination_condition = CONVERGENCE,     \
                plus_phase_termination_criterion=.01,               \
                convergence_function=Distance(metric=MAX_ABS_DIFF), \
                max_passes=None,                                    \
                enable_learning=False,                              \
                learning_rate=None,                                 \
                learning_function=ContrastiveHebbian,               \
                additional_input_states=None,                       \
                additional_output_states=None,                      \
                params=None,                                        \
                name=None,                                          \
                prefs=None)

    Subclass of `RecurrentTransferMechanism` that implements a single-layer auto-recurrent network using two-phases
    of execution and the `Contrastive Hebbian Learning algorithm
    <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_

    COMMENT:
        Description
        -----------
            ContrastiveHebbianMechanism is a Subtype of RecurrentTransferMechanism customized to implement a
            the `ContrastiveHebbian` `LearningFunction <LearningFunctions>`.
    COMMENT

    Arguments
    ---------

    input_size : int : default 0
        specifies the size of the *INPUT* `InputState <ContrastiveHebbian_Input>` and number of units in the
        `input_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`.

    hidden_size : int : default None
        specifies the number of units in the `hidden_field <ContrastiveHebbian_Fields>` of `current_activity
        <ContrastiveHebbianMechanism.current_activity>`.

    target_size : int : default None
        specifies the size of the *TARGET* `InputState <ContrastiveHebbian_Input>`, number of units in the
        `input_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`,
        and of the `OUTPUT_ACTIVITY_OUTPUT` `OutputState <ContrastiveHebbian_Output>`.

    separated : bool : default True
        specifies whether `target_field <ContrastiveHebbian_Fields>` should be different from the `input_field
        <ContrastiveHebbian_Fields>`.

    mode : SIMPLE_HEBBIAN or None : default None
        specifies configuratoin that emulates standard `RecurrentTransferMechanism` using `Hebbian` learning
        (see `SIMPLE_HEBBIAN mode <ContrastiveHebbian_SIMPLE_HEBBIAN>` for details).

    combination_function : function : default None
        specifies function used to combine the *INPUT*, *RECURRENT* and *TARGET* (if specified) `InputStates
        <ContrastiveHebbian_Input>`; must accept a 2d array with two or three items of the same length, and generate a
        result that is the same size as `recurrent_size <ContrastiveHebbianMechanism.recurrent_size>`;  if `None`,
        the ContrastiveHebbianMechanism's combination_function method is used.

    clamp : HARD_CLAMP or SOFT_CLAMP : default HARD_CLAMP
        specifies the manner in which the `value <InputState.value>` of the  *INPUT* and *TARGET* (if specified)
        `InputStates <ContrastiveHebbian_Input>` are combined with the *RECURRENT* `InputState
        <ContrastiveHebbian_Input>` in each execution (see `clamp <ContrastiveHebbianMechanism.clamp>` for additional
        details.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be any function that takes and returns a 1d array
        of scalar values.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`;
        see **matrix** argument of `RecurrentTransferMechanism` for details of specification.

    auto : number, 1D array, or None : default None
        specifies matrix with diagonal entries equal to **auto**; see **auto** argument of
        `RecurrentTransferMechanism` for details of specification.

    hetero : number, 2D array, or None : default None
        specifies a hollow matrix with all non-diagonal entries equal to **hetero**;  see **hetero** argument of
        `RecurrentTransferMechanism` for details of specification.

    continuous : bool : default True
        specifies whether or not to reinitialize `current_activity <ContrastiveHebbianMechanism.current_activity>`
        at the beginning of the `minus phase <ContrastiveHebbian_Minus_Phase>` of a trial.

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use in `integration_mode <ContrastiveHebbianMechanism.integration_mode>`.

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

    minus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        specifies the type of condition used to terminate the `minus_phase <ContrastiveHebbian_Minus_Phase>` of
        execution (see `minus_phase_termination_condition
        <ContrastiveHebbianMechanism.minus_phase_termination_condition>` for additional details).

    minus_phase_termination_criterion : float : default 0.01
        specifies the value of `delta <ContrastiveHebbianMechanism.delta>`
        used to determine when the `minus_phase <ContrastiveHebbian_Minus_Phase>` completes.

    plus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        specifies the type of condition used to terminate the `plus_phase <ContrastiveHebbian_Plus_Phase>` of
        execution (see `plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`
        for additional details).

    plus_phase_termination_criterion : float : default 0.01
        specifies the value of `delta <ContrastiveHebbianMechanism.delta>`
        used to determine when the `plus_phase <ContrastiveHebbian_Plus_Phase>` completes.

    convergence_function : function : default Distance(metric=MAX_ABS_DIFF)
        specifies the function that determines when a `phase of execution <ContrastiveHebbian_Execution>` complete
        if the termination condition for that phase is specified as *CONVERGENCE*, by comparing `current_activity
        <ContrastiveHebbianMechanism.current_activity>` with the `previous_value
        <ContrastiveHebbianMechanism.previous_value>` of the Mechanism;  can be any function that takes two 1d arrays
        of the same length as `variable <ContrastiveHebbianMechanism.variable>` and returns a scalar value. The default
        is the `Distance` Function, using the `MAX_ABS_DIFF` metric  which computes the elementwise difference between
        two arrays and returns the difference with the maximum absolute value.

    max_passes : int : default 1000
        specifies maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the `convergence_criterion
        <ContrastiveHebbianMechanism.convergence_criterion>`, after which an error occurs; if `None` is specified,
        execution may continue indefinitely or until an interpreter exception is generated.

    enable_learning : boolean : default False
        specifies whether the Mechanism should be `configured for learning <ContrastiveHebbian_Learning>`.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <ContrastiveHebbianMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <ContrastiveHebbianMechanism.learning_rate>` for details).

    learning_function : function : default ContrastiveHebbian
        specifies the function for the LearningMechanism if `learning is specified <ContrastiveHebbian_Learning>`.
        It can be any function so long as it takes a list or 1d array of numeric values as its `variable
        <Function_Base.variable>` and returns a square matrix of numeric values with the same dimensions as the
        length of the input.

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

    input_size : int
        size of the *INPUT* `InputState <ContrastiveHebbian_Input>` and `input_activity
        <ContrastiveHebbianMechanism.input_activity>`, and the number of units in the `input_field of
        <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`.

    hidden_size : int
        size of `hidden_activity <ContrastiveHebbianMechanism.input_activity>`, and number of units in
        `current_activity <ContrastiveHebbianMechanism.current_activity>` and the `hidden_field
        <ContrastiveHebbian_Fields>` of the *RECURRENT* `InputState <ContrastiveHebbian_Input>`.

    target_size : int
        size of the *TARGET* `InputState <ContrastiveHebbian_Input>` `if specified <ContrastiveHebbian_Creation>` and,
        if so, the number of units in `target_activity <ContrastiveHebbianMechanism.target_activity>`, the
        `target_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`,
        and the *OUTPUT_ACTIVITY_OUTPUT* `OutputState <ContrastiveHebbian_Output>`.

    target_start : int
        index of first unit of `target_field <ContrastiveHebbian_Fields>`.

    target_end : int
        index of first unit *after* `target_field <ContrastiveHebbian_Fields>`.

    separated : bool : default True
        determines whether `target_field <ContrastiveHebbian_Fields>` is different from `input_field
        <ContrastiveHebbian_Fields>` (`True`) or the same (`False`).

    recurrent_size : int
        size of *RECURRENT* `InputState <ContrastiveHebbian_Input>`, `current_activity
        <ContrastiveHebbianMechanism.current_activity>`, and *CURRENT_ACTIVITY_OUTPUT* `OutputState
        <ContrastiveHebbian_Output>`.

    mode : SIMPLE_HEBBIAN or None
        indicates whether *SIMPLE_HEBBIAN* was used for configuration (see
        `SIMPLE_HEBBIAN mode <ContrastiveHebbian_SIMPLE_HEBBIAN>` for details).

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the *CURRENT_ACTIVITY_OUTPUT* `OutputState
        <ContrastiveHebbian_Output>` to the *RECURRENT* `InputState <ContrastiveHebbian_Input>`.

    variable : value
        the input to Mechanism's `function <ContrastiveHebbianMechanism.variable>`.

    combination_function : method or function
        used to combine `value <InputState.value>` of the *INPUT* and *TARGET* (if specified)  `InputStates
        <ContrastiveHebbian_Input>` with that of the *RECURRENT* `InputState <ContrastiveHebbian_Input>` to determine
        the `variable <CurrentHebbianMechanism.variable>` passed to the Mechanism's `integrator_function
        <ContrastiveHebbianMechanism.integrator_function>` and/or its `function <ContrastiveHebbianMechanism.function>`
        (see `ContrastiveHebbian_Execution` for details).

    clamp : HARD_CLAMP or SOFT_CLAMP
        determines whether the `value <InputState.value>` of the *INPUT* and *TARGET* (if specified) `InputStates
        <ContrastiveHebbian_Input>` replace (*HARD_CLAMP*) or are added to (*SOFT_CLAMP*) the `value <InputState.value>`
        of the *RECURRENT* InputState by `combination_function <ContrastiveHebbianMechanism.combination_function>`.

    continuous : bool : default True
        determines whether or not `current_activity <ContrastiveHebbianMechanism.current_activity>` is reinitialized
        at the beginning of the `minus phase <ContrastiveHebbian_Minus_Phase>` of execution. If `False`, it (and
        the Mechanism's `previous_value <ContrastiveHebbianMechanism.previous_value>` attribute) are set to
        `initial_value <ContrastiveHebbianMechanism.initial_value>`.

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <ContrastiveHebbianMechanism.integrator_mode>` for details).

        .. note::
            The ContrastiveHebbianMechanism's `integration_rate <ContrastiveHebbianMechanism.integration_rate>`, `noise
            <ContrastiveHebbianMechanism.noise>`, and `initial_value <ContrastiveHebbianMechanism.initial_value>`
            parameters specify the respective parameters of its `integrator_function` (with **initial_value**
            corresponding to `initializer <IntegratorFunction.initializer>` of integrator_function.

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
        of the ContrastiveHebbianMechanism as the `rate <IntegratorFunction.rate>` of the ContrastiveHebbianMechanism's
        `integrator_function`; see TransferMechanism's `integrator_function <TransferMechanism.integrator_function>`
        and `ContrastiveHebbian_Execution` above for additional details).

    integration_rate : float
        the rate used for exponential time averaging of input when `integrator_mode
        <ContrastiveHebbianMechanism.integrator_mode>` is set to `True`;  see TransferMechanism's
        `integration_rate <TransferMechanism.integration_rate>` for additional details.

    function : Function
        used to transform the input and generate the Mechanism's `value <ContrastiveHebbianMechanism.value>`
        (see `ContrastiveHebbian_Execution` for additional details).

    clip : list [float, float]
        determines the allowable range for the result of `function <ContrastiveHebbianMechanism.function>`
        see TransferMechanism's `clip <TransferMechanism.clip>` for additional details.

    current_activity : 1d array of floats
        the value of the actvity of the ContrastiveHebbianMechanism following its last `execution
        <ContrastiveHebbian_Execution>`.

    input_activity : 1d array of floats
        value of units in the `input_field <ContrastiveHebbian_Fields>` of `current_activity
        <ContrastiveHebbianMechanism.current_activity>`.

    hidden_activity : 1d array of floats or None
        value of units in the `hidden_field <ContrastiveHebbian_Fields>` of `current_activity
        <ContrastiveHebbianMechanism.current_activity>`;  `None` if hidden_size = 0 or mode = `SIMPLE_HEBBIAN
        <ContrastiveHebbian_SIMPLE_HEBBIAN>`.

    target_activity : 1d array of floats or None
        value of units in the `target_field <ContrastiveHebbian_Fields>` of `current_activity
        <ContrastiveHebbianMechanism.current_activity>` if *TARGET* `InputState is specified
        <ContrastiveHebbian_Input>`. Same as `input_activity <ContrastiveHebbianMechanism.input_activity>`
        if `separated <ContrastiveHebbianMechanism.separated>` is `False`;  `None` if target_size = 0 or mode =
        `SIMPLE_HEBBIAN <ContrastiveHebbian_SIMPLE_HEBBIAN>`.

    recurrent_activity : 1d array of floats
        same as `current_activity <ContrastiveHebbianMechanism.current_activity>`

    minus_phase_activity : 1d array of floats
        the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>` at the end of the
        `minus phase of execution <ContrastiveHebbian_Minus_Phase>`.

    plus_phase_activity : 1d array of floats
        the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>` at the end of the
        `plus phase of execution <ContrastiveHebbian_Plus_Phase>`.

    previous_value : 1d array of floats
        the value of `current_activity <ContrastiveHebbianMechanism.current_activity>` on the `previous
        execution in the current phase <ContrastiveHebbian_Execution>`.

    delta : scalar
        value returned by `convergence_function <RecurrentTransferMechanism.convergence_function>`;  used to determined
        when `is_converged <RecurrentTransferMechanism.is_converged>` is `True`.

    is_converged : bool
        `True` when `delta <ContrastiveHebbianMechanism.delta>` is less than or equal to the
        `minus_phase_termination_criterion <ContrastiveHebbianMechanism.minus_phase_termination_criterion>` in the
        `minus_phase <ContrastiveHebbian_Minus_Phase>`) or `plus_phase_termination_criterion
        <ContrastiveHebbianMechanism.plus_phase_termination_criterion>` in the `minus_phase
        <ContrastiveHebbian_Minus_Phase>`).

    convergence_function : function
        compares the value of `current_activity <ContrastiveHebbianMechanism.current_activity>` with `previous_value
        <ContrastiveHebbianMechanism.previous_value>`; result is assigned as the value of `delta
        <ContrastiveHebbianMechanism.delta>.  Used to determine when a `phase of execution
        <ContrastiveHebbian_Execution>` is complete if the termination condition for that phase is specified as
        *CONVERGENCE*.

    minus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        determines the type of condition used to terminate the `minus_phase <ContrastiveHebbian_Minus_Phase>` of
        execution.  If it is *COUNT*, the Mechanism is executed the number of times specified in
        `minus_phase_termination_criterion <ContrastiveHebbianMechanism.minus_phase_termination_criterion>`;
        if it is *CONVERGENCE*, execution continues until the value returned by `convergence_function
        <ContrastiveHebbianMechanism.convergence_function>` is less than or equal to
        `minus_phase_termination_criterion <ContrastiveHebbianMechanism.minus_phase_termination_criterion>`.

    minus_phase_termination_criterion : float
        the value used for the specified `minus_phase_termination_condition
        <ContrastiveHebbianMechanism.minus_phase_termination_condition>` to determine when the `minus phase of execution
        <ContrastiveHebbian_Minus_Phase>` completes.

    plus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        determines the type of condition used to terminate the `plus_phase <ContrastiveHebbian_Plus_Phase>` of
        execution.  If it is *COUNT*, the Mechanism is executed the number of times specified in
        `plus_phase_termination_criterion <ContrastiveHebbianMechanism.plus_phase_termination_criterion>`;
        if it is *CONVERGENCE*, execution continues until the value returned by `convergence_function
        <ContrastiveHebbianMechanism.convergence_function>` is less than or equal to
        `plus_phase_termination_criterion <ContrastiveHebbianMechanism.plus_phase_termination_criterion>`.

    plus_phase_termination_criterion : float
        the value used for the specified `plus_phase_termination_condition
        <ContrastiveHebbianMechanism.plus_phase_termination_condition>` to determine when the `plus phase of execution
        <ContrastiveHebbian_Plus_Phase>` completes.

    max_passes : int or None
        determines the maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the corresponding convergence_criterion, after which an error
        occurs; if `None` is specified, execution may continue indefinitely or until an interpreter exception is
        generated.

    execution_phase : bool
        indicates current `phase of execution <ContrastiveHebbian_Execution>`.
        `False` = `minus phase <ContrastiveHebbian_Minus_Phase>`;
        `True` = `plus phase <ContrastiveHebbian_Plus_Phase>`.

    learning_enabled : bool
        indicates whether `learning is enabled <ContrastiveHebbian_Learning>`;  see `learning_enabled
        <RecurrentTransferMechanism.learning_enabled>` of RecurrentTransferMechanism for additional details.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <ContrastiveHebbian_Learning>`, and used to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values
        determines the learning rate used by the `learning_function <ContrastiveHebbianMechanism.learning_function>`
        of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignment).

    learning_function : function
        the function used by the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>` if `learning is configured
        <ContrastiveHebbian_Learning>`;  default is `ContrastiveHebbian`.

    value : 2d np.array
        result of executing `function <ContrastiveHebbianMechanism.function>`; same value as first item of
        `output_values <ContrastiveHebbianMechanism.output_values>`.

    output_states : Dict[str: OutputState]
        an OrderedDict with the following `OutputStates <OutputState>` by default:

        * *OUTPUT_ACTIVITY_OUTPUT*, the  `primary OutputState  <OutputState_Primary>` of the Mechanism, the `value
          <OutputState.value>` of which is `target_activity <ContrastiveHebbianMechanism.target_activity>` if a
          *TARGET* `InputState <ContrastiveHebbian_Input>` is implemented;  otherwise,  `input_activity
          <ContrastiveHebbianMechanism.input_activity>` is assigned as its `value <OutputState.value>`.

        * *CURRENT_ACTIVITY_OUTPUT* -- the `value <OutputState.value>` of which is a 1d array containing the activity
          of the ContrastiveHebbianMechanism after each execution;  at the end of an `execution sequence
          <ContrastiveHebbian_Execution>`, it is assigned the value of `plus_phase_activity
          <ContrastiveHebbianMechanism.plus_phase_activity>`.

        * *ACTIVITY_DIFFERENCE_OUTPUT*, the `value <OutputState.value>` of which is a 1d array with the elementwise
          differences in activity between the plus and minus phases at the end of an `execution sequence
          <ContrastiveHebbian_Execution>`.

    output_values : List[1d np.array]
        a list with the `value <OutputState.value>` of each `OutputState` in `output_states
        <ContrastiveHebbianMechanism.output_states>`.

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

    class Parameters(RecurrentTransferMechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ContrastiveHebbianMechanism.variable>`

                    :default value: numpy.array([[0, 0]])
                    :type: numpy.ndarray

                clamp
                    see `clamp <ContrastiveHebbianMechanism.clamp>`

                    :default value: `HARD_CLAMP`
                    :type: str

                combination_function
                    see `combination_function <ContrastiveHebbianMechanism.combination_function>`

                    :default value: None
                    :type:

                continuous
                    see `continuous <ContrastiveHebbianMechanism.continuous>`

                    :default value: True
                    :type: bool

                current_activity
                    see `current_activity <ContrastiveHebbianMechanism.current_activity>`

                    :default value: None
                    :type:

                current_termination_condition
                    see `current_termination_condition <ContrastiveHebbianMechanism.current_termination_condition>`

                    :default value: None
                    :type:

                current_termination_criterion
                    see `current_termination_criterion <ContrastiveHebbianMechanism.current_termination_criterion>`

                    :default value: None
                    :type:

                execution_phase
                    see `execution_phase <ContrastiveHebbianMechanism.execution_phase>`

                    :default value: None
                    :type:
                    :read only: True

                hidden_activity
                    see `hidden_activity <ContrastiveHebbianMechanism.hidden_activity>`

                    :default value: None
                    :type:
                    :read only: True

                hidden_size
                    see `hidden_size <ContrastiveHebbianMechanism.hidden_size>`

                    :default value: None
                    :type:

                input_activity
                    see `input_activity <ContrastiveHebbianMechanism.input_activity>`

                    :default value: None
                    :type:
                    :read only: True

                input_size
                    see `input_size <ContrastiveHebbianMechanism.input_size>`

                    :default value: None
                    :type:

                is_finished_
                    see `is_finished_ <ContrastiveHebbianMechanism.is_finished_>`

                    :default value: False
                    :type: bool
                    :read only: True

                learning_function
                    see `learning_function <ContrastiveHebbianMechanism.learning_function>`

                    :default value: `ContrastiveHebbian`
                    :type: `Function`

                minus_phase_activity
                    see `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`

                    :default value: None
                    :type:

                minus_phase_termination_condition
                    see `minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>`

                    :default value: `CONVERGENCE`
                    :type: str

                minus_phase_termination_criterion
                    see `minus_phase_termination_criterion <ContrastiveHebbianMechanism.minus_phase_termination_criterion>`

                    :default value: 0.01
                    :type: float

                mode
                    see `mode <ContrastiveHebbianMechanism.mode>`

                    :default value: None
                    :type:

                output_activity
                    see `output_activity <ContrastiveHebbianMechanism.output_activity>`

                    :default value: None
                    :type:
                    :read only: True

                phase_execution_count
                    see `phase_execution_count <ContrastiveHebbianMechanism.phase_execution_count>`

                    :default value: 0
                    :type: int

                phase_terminated
                    see `phase_terminated <ContrastiveHebbianMechanism.phase_terminated>`

                    :default value: False
                    :type: bool

                plus_phase_activity
                    see `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>`

                    :default value: None
                    :type:

                plus_phase_termination_condition
                    see `plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`

                    :default value: `CONVERGENCE`
                    :type: str

                plus_phase_termination_criterion
                    see `plus_phase_termination_criterion <ContrastiveHebbianMechanism.plus_phase_termination_criterion>`

                    :default value: 0.01
                    :type: float

                separated
                    see `separated <ContrastiveHebbianMechanism.separated>`

                    :default value: True
                    :type: bool

                target_activity
                    see `target_activity <ContrastiveHebbianMechanism.target_activity>`

                    :default value: None
                    :type:
                    :read only: True

                target_size
                    see `target_size <ContrastiveHebbianMechanism.target_size>`

                    :default value: None
                    :type:

        """
        variable = np.array([[0, 0]])
        current_activity = Parameter(None, aliases=['recurrent_activity'])
        plus_phase_activity = None
        minus_phase_activity = None
        current_termination_condition = None
        current_termination_criterion = None
        phase_execution_count = 0
        phase_terminated = False

        input_size = Parameter(None, stateful=False, loggable=False)
        hidden_size = Parameter(None, stateful=False, loggable=False)
        target_size = Parameter(None, stateful=False, loggable=False)
        separated = Parameter(True, stateful=False, loggable=False)
        mode = Parameter(None, stateful=False, loggable=False)
        continuous = Parameter(True, stateful=False, loggable=False)
        clamp = Parameter(HARD_CLAMP, stateful=False, loggable=False)
        combination_function = Parameter(None, stateful=False, loggable=False)
        minus_phase_termination_condition = Parameter(CONVERGENCE, stateful=False, loggable=False)
        plus_phase_termination_condition = Parameter(CONVERGENCE, stateful=False, loggable=False)
        learning_function = Parameter(ContrastiveHebbian, stateful=False, loggable=False)

        output_activity = Parameter(None, read_only=True, getter=_CHM_output_activity_getter)
        input_activity = Parameter(None, read_only=True, getter=_CHM_input_activity_getter)
        hidden_activity = Parameter(None, read_only=True, getter=_CHM_hidden_activity_getter)
        target_activity = Parameter(None, read_only=True, getter=_CHM_target_activity_getter)

        execution_phase = Parameter(None, read_only=True)
        is_finished_ = Parameter(False, read_only=True)

        minus_phase_termination_criterion = Parameter(0.01, modulable=True)
        plus_phase_termination_criterion = Parameter(0.01, modulable=True)

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:OUTPUT_ACTIVITY_OUTPUT,
                                    VARIABLE:OUTPUT_ACTIVITY},
                                   {NAME:CURRENT_ACTIVITY_OUTPUT,
                                    VARIABLE:CURRENT_ACTIVITY},
                                   {NAME:ACTIVITY_DIFFERENCE_OUTPUT,
                                    VARIABLE:[PLUS_PHASE_ACTIVITY, MINUS_PHASE_ACTIVITY],
                                    FUNCTION: lambda v: v[0] - v[1]},
                                   {NAME:MINUS_PHASE_OUTPUT,
                                    VARIABLE:MINUS_PHASE_ACTIVITY},
                                   {NAME:PLUS_PHASE_OUTPUT,
                                    VARIABLE:PLUS_PHASE_ACTIVITY},
                                   ])

    @tc.typecheck
    def __init__(self,
                 input_size:int,
                 hidden_size:tc.optional(int)=None,
                 target_size:tc.optional(int)=None,
                 separated:bool=True,
                 mode:tc.optional(tc.enum(SIMPLE_HEBBIAN))=None,
                 continuous:bool=True,
                 clamp:tc.enum(SOFT_CLAMP, HARD_CLAMP)=HARD_CLAMP,
                 combination_function:tc.optional(is_function_type)=None,
                 function=Linear,
                 matrix=HOLLOW_MATRIX,
                 auto=None,
                 hetero=None,
                 integrator_function=AdaptiveIntegrator,
                 initial_value=None,
                 noise=0.0,
                 integration_rate: is_numeric_or_none=0.5,
                 integrator_mode:bool=False,
                 clip=None,
                 minus_phase_termination_condition:tc.enum(CONVERGENCE, COUNT)=CONVERGENCE,
                 minus_phase_termination_criterion:float=0.01,
                 plus_phase_termination_condition:tc.enum(CONVERGENCE, COUNT)=CONVERGENCE,
                 plus_phase_termination_criterion:float=0.01,
                 convergence_function:tc.any(is_function_type)=Distance(metric=MAX_ABS_DIFF),
                 max_passes:tc.optional(int)=1000,
                 enable_learning:bool=False,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = ContrastiveHebbian,
                 additional_input_states:tc.optional(tc.any(list, dict)) = None,
                 additional_output_states:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None,
                 **kwargs):
        """Instantiate ContrastiveHebbianMechanism
        """

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self, self.standard_output_states, indices=PRIMARY)

        if mode is SIMPLE_HEBBIAN:
            hidden_size=0
            # target_size=0
            separated = False
            clamp = SOFT_CLAMP
            continuous = False
            learning_function = Hebbian

        self.input_size = input_size
        self.hidden_size = hidden_size or 0
        self.target_size = target_size or 0
        self.separated = separated
        self.recurrent_size = input_size + hidden_size
        if separated and target_size:
            self.recurrent_size += target_size
            self.target_start = input_size + hidden_size
            self._target_included = True
        else:
            self.target_start = 0
            self._target_included = False
        self.target_end = self.target_start + self.target_size
        size = self.recurrent_size

        default_variable = [np.zeros(input_size), np.zeros(self.recurrent_size)]
        # Set InputState sizes in _instantiate_input_states,
        #    so that there is no conflict with parsing of Mechanism's size
        input_states = [INPUT, RECURRENT]
        if self._target_included:
            default_variable.append(np.zeros(target_size))
            input_states.append(TARGET)

        if additional_input_states:
            if isinstance(additional_input_states, list):
                input_states += additional_input_states
            else:
                input_states.append(additional_input_states)

        combination_function = combination_function or self.combination_function

        output_states = [OUTPUT_ACTIVITY_OUTPUT, CURRENT_ACTIVITY_OUTPUT, ACTIVITY_DIFFERENCE_OUTPUT]
        if additional_output_states:
            if isinstance(additional_output_states, list):
                output_states += additional_output_states
            else:
                output_states.append(additional_output_states)

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(mode=mode,
                                                  minus_phase_termination_condition=minus_phase_termination_condition,
                                                  minus_phase_termination_criterion=minus_phase_termination_criterion,
                                                  plus_phase_termination_condition=plus_phase_termination_condition,
                                                  plus_phase_termination_criterion=plus_phase_termination_criterion,
                                                  continuous=continuous,
                                                  clamp=clamp,
                                                  input_states=input_states,
                                                  output_states=output_states,
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
                         integrator_function=integrator_function,
                         initial_value=initial_value,
                         noise=noise,
                         integrator_mode=integrator_mode,
                         integration_rate=integration_rate,
                         clip=clip,
                         convergence_function=convergence_function,
                         convergence_criterion=minus_phase_termination_criterion,
                         max_passes=max_passes,
                         enable_learning=enable_learning,
                         learning_rate=learning_rate,
                         learning_function=learning_function,
                         learning_condition=CONVERGENCE,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Make sure that the size of the INPUT and TARGET InputStates are <= size of RECURRENT InputState
        if self._target_included and self.input_size != self.target_size:
            raise ContrastiveHebbianError("{} is {} for {} must equal {} ({}) must equal {} ({})} ".
                                          format(repr(SEPARATED), repr(True), self.name,
                                                 repr(INPUT_SIZE), self.input_size,
                                                 repr(TARGET_SIZE), self.target_size))


    def _instantiate_input_states(self, input_states=None, reference_value=None, context=None):

        # Assign InputState specification dictionaries for required InputStates
        sizes = dict(INPUT=self.input_size, RECURRENT=self.recurrent_size, TARGET=self.target_size)
        for i, input_state in enumerate((s for s in self.input_states if s in {INPUT, TARGET, RECURRENT})):
            self.input_states[i] = {NAME:input_state, SIZE: sizes[input_state]}

        super()._instantiate_input_states(input_states, reference_value, context)

        self.input_states[RECURRENT].internal_only = True
        if self._target_included:
            self.input_states[TARGET].internal_only = True

    @tc.typecheck
    def _instantiate_recurrent_projection(self,
                                          mech: Mechanism,
                                          # this typecheck was failing, I didn't want to fix (7/19/17 CW)
                                          # matrix:is_matrix=HOLLOW_MATRIX,
                                          matrix=HOLLOW_MATRIX,
                                          context=None):
        """Instantiate an AutoAssociativeProjection from Mechanism to itself
        """

        from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection
        if isinstance(matrix, str):
            size = len(mech.defaults.variable[0])
            matrix = get_matrix(matrix, size, size)

        return AutoAssociativeProjection(owner=mech,
                                         sender=self.output_states[CURRENT_ACTIVITY_OUTPUT],
                                         receiver=self.input_states[RECURRENT],
                                         matrix=matrix,
                                         name=mech.name + ' recurrent projection')

    def _instantiate_attributes_after_function(self, context=None):

        # Assign these after instantiation of function, since they are initialized in _execute (see below)
        self.attributes_dict_entries.update({OUTPUT_ACTIVITY:OUTPUT_ACTIVITY,
                                             CURRENT_ACTIVITY:CURRENT_ACTIVITY,
                                             MINUS_PHASE_ACTIVITY:MINUS_PHASE_ACTIVITY,
                                             PLUS_PHASE_ACTIVITY:PLUS_PHASE_ACTIVITY})
        super()._instantiate_attributes_after_function(context=context)

    def _execute(self,
                 variable=None,
                 context=None,
                 function_variable=None,
                 runtime_params=None,
                 ):

        if self.initialization_status == ContextFlags.INITIALIZING:
            # Set minus_phase activity, plus_phase, current_activity and initial_value
            #    all  to zeros with size of Mechanism's array
            # Should be OK to use attributes here because initialization should only occur during None context
            self._set_multiple_parameter_values(
                context,
                initial_value=self.input_states[RECURRENT].socket_template,
                current_activity=self.input_states[RECURRENT].socket_template,
                minus_phase_activity=self.input_states[RECURRENT].socket_template,
                plus_phase_activity=self.input_states[RECURRENT].socket_template,
                execution_phase=None,
            )
            if self._target_included:
                self.parameters.output_activity._set(self.input_states[TARGET].socket_template, context)

        # Initialize execution_phase as minus_phase
        if self.parameters.execution_phase._get(context) is None:
            self.parameters.execution_phase._set(MINUS_PHASE, context)

        if self.parameters.execution_phase._get(context) is MINUS_PHASE:
            self.parameters.current_termination_criterion._set(self.parameters.minus_phase_termination_criterion._get(context), context)
            self.parameters.current_termination_condition._set(self.minus_phase_termination_condition, context)
            self.parameters.phase_execution_count._set(0, context)

        if self.parameters.is_finished_._get(context):
            # If current execution follows completion of a previous trial,
            #    zero activity for input from recurrent projection so that
            #    input does not contain residual activity of previous trial
            variable[RECURRENT_INDEX] = self.input_states[RECURRENT].socket_template

        self.parameters.is_finished_._set(False, context)

        # Need to store this, as it will be updated in call to super
        previous_value = self.parameters.previous_value._get(context)

        # Note _parse_function_variable selects actual input to function based on execution_phase
        current_activity = super()._execute(variable,
                                            context=context,
                                            runtime_params=runtime_params,
                                            )

        self.parameters.phase_execution_count._set(self.parameters.phase_execution_count._get(context) + 1, context)

        current_activity = np.squeeze(current_activity)
        # Set value of primary OutputState to current activity
        self.parameters.current_activity._set(current_activity, context)

        # This is the first trial, so can't test for convergence
        #    (since that requires comparison with value from previous trial)
        if previous_value is None:
            return current_activity

        current_termination_condition = self.parameters.current_termination_condition._get(context)

        if current_termination_condition is CONVERGENCE:
            self.parameters.convergence_criterion._set(
                self.parameters.current_termination_criterion._get(context),
                context
            )
            self.parameters.phase_terminated._set(self.is_converged(np.atleast_2d(current_activity), context), context)
        elif current_termination_condition is COUNT:
            self.parameters.phase_terminated._set(
                (self.parameters.phase_execution_count._get(context) == self.parameters.current_termination_criterion._get(context)),
                context
            )
        else:
            raise ContrastiveHebbianError(
                "Unrecognized {} specification ({}) in execution of {} of {}".format(
                    repr('current_termination_condition'),
                    current_termination_condition,
                    repr('PLUS_PHASE') if self.parameters.execution_phase._get(context) is PLUS_PHASE else repr('MINUS_PHASE'),
                    self.name
                )
            )

        if self.parameters.phase_terminated._get(context):
            # Terminate if this is the end of the plus phase, prepare for next trial
            if self.parameters.execution_phase._get(context) == PLUS_PHASE:
                # Store activity from last execution in plus phase
                self.parameters.plus_phase_activity._set(current_activity, context)
                # # Set value of primary outputState to activity at end of plus phase
                # self.current_activity = self.plus_phase_activity
                self.parameters.current_activity._set(current_activity, context)
                # self.execution_phase = None
                self.parameters.is_finished_._set(True, context)

            # Otherwise, prepare for start of plus phase on next execution
            else:
                # Store activity from last execution in plus phase
                self.parameters.minus_phase_activity._set(self.parameters.current_activity._get(context), context)
                # Use initial_value attribute to initialize, for the minus phase,
                #    both the integrator_function's previous_value
                #    and the Mechanism's current activity (which is returned as its input)
                if not self.continuous:
                    self.reinitialize(self.initial_value, context=context)
                    self.parameters.current_activity._set(self.parameters.initial_value._get(context), context)
                self.parameters.current_termination_criterion._set(self.plus_phase_termination_criterion, context)
                self.parameters.current_termination_condition._set(self.plus_phase_termination_condition, context)

            # Switch execution_phase
            self.parameters.execution_phase._set(not self.parameters.execution_phase._get(context), context)
            self.parameters.phase_execution_count._set(0, context)

        return current_activity
        # return self.current_activity

    def _parse_function_variable(self, variable, context=None):
        function_variable = self.combination_function(variable=variable, context=context)
        return super(RecurrentTransferMechanism, self)._parse_function_variable(function_variable, context=context)

    def combination_function(self, variable=None, context=None):
        # IMPLEMENTATION NOTE: use try and except here for efficiency: care more about execution than initialization
        # IMPLEMENTATION NOTE: separated vs. overlapping input and target handled by assignment of target_start in init

        MINUS_PHASE_INDEX = INPUT_INDEX
        PLUS_PHASE_INDEX = TARGET_INDEX

        try:  # Execution
            if self.parameters.execution_phase._get(context) == PLUS_PHASE:
                if self.clamp == HARD_CLAMP:
                    variable[RECURRENT_INDEX][:self.input_size] = variable[MINUS_PHASE_INDEX]
                    if self.mode is SIMPLE_HEBBIAN:
                        return variable[RECURRENT_INDEX]
                    else:
                        variable[RECURRENT_INDEX][self.target_start:self.target_end] = variable[PLUS_PHASE_INDEX]
                else:
                    variable[RECURRENT_INDEX][:self.input_size] += variable[MINUS_PHASE_INDEX]
                    if self.mode is SIMPLE_HEBBIAN:
                        return variable[RECURRENT_INDEX]
                    else:
                        variable[RECURRENT_INDEX][self.target_start:self.target_end] += variable[PLUS_PHASE_INDEX]
            else:
                if self.mode is SIMPLE_HEBBIAN:
                    return variable[RECURRENT_INDEX]
                if self.clamp == HARD_CLAMP:
                    variable[RECURRENT_INDEX][:self.input_size] = variable[MINUS_PHASE_INDEX]
                else:
                    variable[RECURRENT_INDEX][:self.input_size] += variable[MINUS_PHASE_INDEX]
        except:  # Initialization
            pass

        return variable[RECURRENT_INDEX]

    @property
    def _learning_signal_source(self):
        """Override default to use ACTIVITY_DIFFERENCE_OUTPUT as source of learning signal
        """
        return self.output_states[ACTIVITY_DIFFERENCE_OUTPUT]

    @property
    def input_activity(self):
        return self.current_activity[:self.input_size]

    @property
    def hidden_activity(self):
        if self.hidden_size:
            return self.current_activity[self.input_size:self.target_start]

    @property
    def target_activity(self):
        if self.target_size:
            return self.current_activity[self.target_start:self.target_end]

    @property
    def recurrent_activity(self):
        return self.current_activity

    def is_finished(self, context=None):
        # is a method, to be compatible with scheduling
        return self.parameters.is_finished_.get(context)
