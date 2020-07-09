# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** ContrastiveHebbianMechanism *************************************************

"""

Contents
--------

  * `ContrastiveHebbian_Overview`
  * `ContrastiveHebbian_Creation`
      - `ContrastiveHebbian_Size`
      - `ContrastiveHebbian_Phases`
      - `ContrastiveHebbian_OutputPorts`
      - `ContrastiveHebbian_Learning`
      - `ContrastiveHebbian_SIMPLE_HEBBIAN`
  * `ContrastiveHebbian_Structure`
      - `ContrastiveHebbian_Input`
      - `ContrastiveHebbian_Functions`
      - `ContrastiveHebbian_Output`
  * `ContrastiveHebbian_Execution`
      - `ContrastiveHebbian_Processing`
      - `ContrastiveHebbian_Learning_Execution`
  * `ContrastiveHebbian_Class_Reference`


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

Creating a ContrastiveHebbianMechanism
--------------------------------------

.. _ContrastiveHebbian_Size:

*Size*
~~~~~~

The **input_size** argument of the constructor must always be specified (this is comparable to specifying the
**size** or *default_variable** arguments of other types of `Mechanism`).  If it is specified on its own,
it determines the total number of processing units.  If either the **hidden_size** and/or **target_size** arguments
are specified, then those units are treated as distinct from the input units (see `ContrastiveHebbian_Execution` for
details).

.. _ContrastiveHebbian_Phases:

*Phases*
~~~~~~~~

A ContrastiveHebbianMechanism `executes in two phases <ContrastiveHebbian_Execution>`, and has
**minus_phase_termination_condition** and **plus_phase_termination_condition** arguments, and corresponding
**minus_phase_termination_threshold** and **plus_phase_termination_threshold** arguments, that determine when the
respective phases of execution are terminated.  Other parameters can also be configured that influence processing (see
`ContrastiveHebbian_Execution`).

.. _ContrastiveHebbian_OutputPorts:

*OututPorts*
~~~~~~~~~~~~

The Mechanism is automatically assigned three of its five `standard_output_ports
<ContrastiveHebbianMechanism.standard_output_ports>`: *OUTPUT_ACTIVITY*, *CURRENT_ACTIVITY*,
and *ACTIVITY_DIFFERENT* (see `below <ContrastiveHebbian_Output>`). Additional OutputPorts can be specified
in the **additional_output_ports** argument of the constructor.

.. _ContrastiveHebbian_Learning:

*Learning*
~~~~~~~~~~

A ContrastiveHebbianMechanism can be configured for learning either by specifying **enable_learning** as `True` or using
the `configure_learning <RecurrentTransferMechanism.configure_learning>` method, with the following differences from a
standard `RecurrentTransferMechanism <RecurrentTransferMechanism_Learning>`:  it is automatically assigned
`ContrastiveHebbian` as its `learning_function <ContrastiveHebbianMechanism.learning_function>`; its `learning_condition
<RecurrentTransferMechanism.learning_condition>` is automatically assigned as *CONVERGENCE*; and it is assigned a
`MappingProjection` from its *ACTIVITY_DIFFERENCE* (rather than its `primary <OutputPort_Primary>`)
`OutputPort <ContrastiveHebbian_Output>` to the *ACTIVATION_INPUT* of its `learning_mechanism
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

A ContrastiveHebbianMechanism always has two, and possibly three `InputPorts <InputPort>`: 

    * *INPUT:* receives external input to the Mechanism;
    ..
    * *RECURRENT:* receives the `value <Projection_Base.value>` of the Mechanism's `recurrent_projection
      <ContrastiveHebbianMechanism.recurrent_projection>`;
    ..
    * *TARGET:* only implemented if **target_size** is specified, **separated = `True` (default), and
      mode is not `SIMPLE_HEBBIAN <ContrastiveHebbian_SIMPLE_HEBBIAN>`;  receives the `target <Run.target>`
      specified in the `run <System.run>` method of any `Composition` to which the Mechanism belongs.

The sizes of these are determined by arguments in its constructor, which generally conform to one of two
configurations.

|    **Standard Configuration** --   if **input_size**, **hidden_size** and **target_size** are all specified,
|    the sizes of the InputPorts are as follows:
|
|        *INPUT*:  size = **input_size**
|        *RECURRENT*:  size = **input_size** + **hidden_size** + **target_size**
|        *TARGET*:  size = **input_size**
|
|    **Simple Configuration** -- if **target_size** = `None` or 0,  **separated** = `False`, and/or **mode** =
|    *SIMPLE_HEBBIAN*, the sizes of the InputPorts are as follows:
|
|        *INPUT*:  size = **input_size**
|        *RECURRENT*:  size = **input_size** + **hidden_size**
|        *TARGET*:  Not implemented
|
.. note::
   If **separated** = `False` and **target_size** is specified, a *TARGET* InputPort will be created, but
   **target_size** must equal **input_size** or an error will be generated.

The values of **input_size**, **hidden_size** and **target_size** are assigned to the Mechanism's
`input_size <ContrastiveHebbianMechanism.input_size>`, `hidden_size <ContrastiveHebbianMechanism.hidden_size>`,
and `target_size <ContrastiveHebbianMechanism.target_size>` attributes, respectively.

.. _ContrastiveHebbian_Fields:

The `input_size <ContrastiveHebbianMechanism.input_size>` and `target_size <ContrastiveHebbianMechanism.target_size>`
(if **separated** is `True`) attribute(s), together with the `separated <ContrastiveHebbianMechanism>` attribute
are used to define the fields of the *RECURRENT* InputPort's `value <InputPort.value>` and the `current_activity
<ContrastiveHebbianMechanism>` attribute used for updating values during `execution <ContrastiveHebbian_Processing>`,
as follows:

    * *input_field:*  the leftmost number of elements determined by `input_size
      <ContrastiveHebbianMechanism.input_size>`;
    ..
    * *hidden_field:*  the elements of the *RECURRENT* InputPort and `current_activity <ContrastiveHebbianMechanism>`
      that are not within the *input_field* and/or *target_field*;
    ..
    * *target_field:* the rightmost number of elements determined by `target_size
      <ContrastiveHebbianMechanism.target_size>` if `separated <ContrastiveHebbianMechanism.separated>` is `True`;
      otherwise, the same as *input_field*.

.. _ContrastiveHebbian_Functions:

*Functions*
~~~~~~~~~~~

In addition to its primary `function <Mechanism_Base.function>`, if either the
`minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>` or
`plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`
is specified as *CONVERGENCE*, then its `phase_phase_convergence_function
<ContrastiveHebbianMechanism.phase_phase_convergence_function>` is used to determine when the corresponding
`phase of execution <ContrastiveHebbian_Execution>` is complete. Its `learning_function
<ContrastiveHebbianMechanism.learning_function>` is automatically assigned as `ContrastiveHebbian`, but this can be
replaced by any function that takes two 1d arrays ("activity ports") and compares them to determine the `matrix
<MappingProjection.matrix>`  of the Mechanism's `recurrent_projection
<ContrastiveHebbianMechanism.recurrent_projection>`.  If **mode** is specified as `SIMPLE_HEBBIAN
<ContrastiveHebbian_SIMPLE_HEBBIAN>`), the default `function <Mechanism_Base.function>` is `Hebbian`,
but can be replaced by any function that takes and returns a 1d array.

.. _ContrastiveHebbian_Output:

*Output*
~~~~~~~~

A ContrastiveHebbianMechanism is automatically assigned three `OutputPorts <OutputPort>`:

* *OUTPUT_ACTIVITY:* assigned as the `primary OutputPort <OutputPort_Primary>`, and contains the pattern
  of activity the Mechanism is trained to generate.  If **target_size** is specified, then it has the same size as
  the *TARGET* InputPort;  if **target_size** is not specified or is 0, if `separated
  <ContrastiveHebbianMechanism.separated>` is `False`, or if mode is *SIMPLE_HEBBIAN*, then the size of the
  *OUTPUT_ACTIVITY* OutputPort is the same as the *INPUT* InputPort (see `ContrastiveHebbian_Input`).
..
* *CURRENT_ACTIVITY:* assigned the value of the `current_activity <ContrastiveHebbianMechanism.current_activity>`
  attribute after each `execution <ContrastiveHebbian_Execution>` of the Mechanism, which contains the activity of all
  processing units in the Mechanism (input, hidden and/or target);  has the same size as the *RECURRENT* InputPort.
..
* *ACTIVITY_DIFFERENCE:* assigned the difference between `plus_phase_activity
  <ContrastiveHebbianMechanism.plus_phase_activity>` and `minus_phase_activity
  <ContrastiveHebbianMechanism.minus_phase_activity>` at the `completion of execution
  <ContrastiveHebbian_Execution>`. If `configured for learning <ContrastiveHebbian_Learning>`, a `MappingProjection`
  is assigned from the *ACTIVITY_DIFFERENCE* `OutputPort <ContrastiveHebbian_Output>` to the
  *ACTIVATION_INPUT* of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>`.

These also appear in the ContrastiveHebbianMechanism's `standard_output_ports
<ContrastiveHebbianMechanism.standard_output_ports>`, along with two others -- *MINUS_PHASE_ACTIVITY* and
*PLUS_PHASE_ACTIVITY* -- that it can be assigned, in addition to the `standard_output_ports
<RecurrentTransferMechanism.standard_output_ports>` of a `RecurrentTransferMechanism`.


.. _ContrastiveHebbian_Execution:

Execution
---------

.. _ContrastiveHebbian_Processing:

*Processing*
~~~~~~~~~~~~

A ContrastiveHebbianMechanism always executes in two sequential phases, that together constitute a *trial of execution:*

.. _ContrastiveHebbian_Minus_Phase:

* *minus phase:* in each execution, the *RECURRENT* InputPort's `value <InputPort.value>` (received from the
  `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`) is combined with the *INPUT*
  InputPort's `value <InputPort.value>`. The result of `function <Mechanism_Base.function>` is assigned to
  `current_activity <ContrastiveHebbianMechanism.current_activity>`.  The Mechanism is executed repeatedly
  until its `minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>`
  is met. At that point, the *minus phase* is completed, the `value <Mechanism_Base.value>` of the
  ContrastiveHebbianMechanism is assigned to its `minus_phase_activity
  <ContrastiveHebbianMechanism.minus_phase_activity>` attribute, and the *plus phase* is begun.
..

.. _ContrastiveHebbian_Plus_Phase:

* *plus phase:*  if `continuous <ContrastiveHebbianMechanism.continuous>` is `False`, then `current_activity
  <ContrastiveHebbianMechanism.current_activity>` is reset to
  `initial_value <Mechanism_Base.initial_value>`, and the Mechanism's
  previous `value <Mechanism_Base.value>` is reset to ``None``;
  otherwise, these retain their value from the last execution in the
  *minus phase*.  In either case, the *RECURRENT* InputPort's `value <InputPort.value>` is combined with the *INPUT*
  InputPort's `value <InputPort.value>` (as during the `minus_phase
  <ContrastiveHebbianMechanism.minus_phase_activity>`) as well as that of *TARGET* InputPort (if that is `specified
  <ContrastiveHebbian_Input>`). If `separated <ContrastiveHebbianMechanism.separated>` is `True` (which it is by
  default), then the `value <InputPort.value>` of the *INPUT* `InputPort <ContrastiveHebbian_Input>` is combined with
  the `input_field <ContrastiveHebbian_Fields>` of the *RECURRENT* InputPort's `value <InputPort.value>`,
  and *TARGET* is combined with the `target_field <ContrastiveHebbian_Fields>`; otherwise, both are combined with the
  `input_field <ContrastiveHebbian_Fields>`.  If `hidden_size <ContrastiveHebbianMechanism.hidden_size>` is
  specified, then the `hidden_field <ContrastiveHebbian_Fields>` of the *RECURRENT* InputPort's  `value
  <InputPort.value>` is determined only by the `value <AutoAssociativeProjection.value>` of the Mechanism's
  `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.  Execution then proceeds as during the
  *minus phase*, completing when its `plus_phase_termination_condition
  <ContrastiveHebbianMechanism.plus_phase_termination_condition>` is met.  At that point, the *plus phase* is
  completed, and the `value <Mechanism_Base.value>` of the Mechanism is assigned to
  `plus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`.

The `value <InputPort.value>` of the *INPUT*, and possibly *TARGET*, InptutState(s) are combined with that of its
*RECURRENT* InputPort using the `combination_function <ContrastiveHebbianMechanism.combination_function>`. The
manner in which these are combined is determined by the `clamp <ContrastiveHebbianMechanism.clamp>` attribute: if
it is *HARD_CLAMP* they are used to replace the corresponding fields of *RECURRENT*;  if it is *SOFT_CLAMP*, *INPUT*
(and possibly *TARGET*) are added to *RECURRENT*; .  The result is passed to the Mechanism's `integrator_function
<TransferMechanism.integrator_function>` (if `integrator_mode <TransferMechanism.integrator_mode>`
is `True`) and then its `function <Mechanism_Base.function>`.

If the termination condition for either phase is specified as *CONVERGENCE*, it uses the Mechanism's
`phase_convergence_function <ContrastiveHebbianMechanism.phase_convergence_function>`, together with the
termination_threshold specified for that phase, to determine when execution of that phase terminates.  If `max_passes
<ContrastiveHebbianMechanism.max_passes>` is specified, and the number of executions in either phase
reaches that value, an error is generated.  Otherwise, once a trial of execution is complete (i.e, after completion
of the *minus phase*), the following computations and assignments are made:

* if a *TARGET* InputPort `has been specified <ContrastiveHebbian_Input>`, then the `target field
  <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>` is assigned as
  `value <OutputPort.value>` of *OUTPUT_ACTIVITY* `OutputPort <ContrastiveHebbian_Output>`;  otherwise,
  it is assigned the value of the `input_field <ContrastiveHebbian_Fields>` of `current_activity
  <ContrastiveHebbianMechanism.current_activity>`.
..
* `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` is assigned as the `value
  <OutputPort.value>` of the *CURRENT_ACTIVITY_ATTR* `OutputPort <ContrastiveHebbian_Output>`;
..
* the difference between `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>` and
  `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>` is assigned as the `value
  <OutputPort.value>` of the *ACTIVITY_DIFFERENCE* `OutputPort <ContrastiveHebbian_Output>`.

.. _ContrastiveHebbian_Learning_Execution:

*Learning*
~~~~~~~~~~

If a ContrastiveHebbianMechanism is `configured for learning <ContrastiveHebbian_Learning>`, at the end of each
`trial of execution <ContrastiveHebbian_Processing>` the `value <OutputPort.value>` of its *ACTIVITY_DIFFERENCE*
`OutputPort <ContrastiveHebbian_Output>` is passed to its `learning_mechanism
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

import copy
import numpy as np
import typecheck as tc
from psyneulink.core.components.functions.function import get_matrix, is_function_type
from psyneulink.core.components.functions.learningfunctions import ContrastiveHebbian, Hebbian
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    CONTRASTIVE_HEBBIAN_MECHANISM, COUNT, FUNCTION, HARD_CLAMP, HOLLOW_MATRIX, MAX_ABS_DIFF, NAME, \
    SIZE, SOFT_CLAMP, TARGET, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    CONVERGENCE, RECURRENT, RECURRENT_INDEX, RecurrentTransferMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

__all__ = [
    'ACTIVITY_DIFFERENCE', 'CURRENT_ACTIVITY', 'SIMPLE_HEBBIAN', 'INPUT',
    'MINUS_PHASE_ACTIVITY', 'PLUS_PHASE_ACTIVITY', 'OUTPUT_ACTIVITY',
    'ContrastiveHebbianError', 'ContrastiveHebbianMechanism',
]

INPUT = 'INPUT'

INPUT_SIZE = 'input_size'
HIDDEN_SIZE = 'hidden_size'
TARGET_SIZE = 'target_size'
SEPARATED = 'separated'

SIMPLE_HEBBIAN = 'SIMPLE_HEBBIAN'

OUTPUT_ACTIVITY = 'OUTPUT_ACTIVITY'
CURRENT_ACTIVITY = 'CURRENT_ACTIVITY'
ACTIVITY_DIFFERENCE = 'ACTIVITY_DIFFERENCE'
MINUS_PHASE_ACTIVITY = 'MINUS_PHASE_ACTIVITY'
PLUS_PHASE_ACTIVITY = 'PLUS_PHASE_ACTIVITY'

OUTPUT_ACTIVITY_ATTR = 'output_activity'
CURRENT_ACTIVITY_ATTR = 'current_activity'
MINUS_PHASE_ACTIVITY_ATTR = 'minus_phase_activity'
PLUS_PHASE_ACTIVITY_ATTR = 'plus_phase_activity'

INPUT_INDEX = 0
RECURRENT_INDEX = 1
TARGET_INDEX = 2

MINUS_PHASE = False
PLUS_PHASE  = True


class ContrastiveHebbianError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _CHM_output_activity_getter(owning_component=None, context=None):
    current_activity = owning_component.parameters.current_activity._get(context)
    if owning_component.target_size:
        return current_activity[owning_component.target_start:owning_component.target_end]
    else:
        return current_activity[:owning_component.input_size]

def _CHM_input_activity_getter(owning_component=None, context=None):
    current_activity = owning_component.parameters.current_activity._get(context)
    return current_activity[:owning_component.input_size]


def _CHM_hidden_activity_getter(owning_component=None, context=None):
    if owning_component.hidden_size:
        current_activity = owning_component.parameters.current_activity._get(context)
        return current_activity[owning_component.input_size:owning_component.target_start]


def _CHM_target_activity_getter(owning_component=None, context=None):
    if owning_component.target_size:
        current_activity = owning_component.parameters.current_activity._get(context)
        return current_activity[owning_component.target_start:owning_component.target_end]


class ContrastiveHebbianMechanism(RecurrentTransferMechanism):
    """
    ContrastiveHebbianMechanism(                                          \
                input_size=2,                                             \
                hidden_size=None,                                         \
                target_size=None,                                         \
                separated:bool=True,                                      \
                mode=None,                                                \
                continuous=True,                                          \
                clamp=HARD_CLAMP,                                         \
                minus_phase_termination_condition = CONVERGENCE,          \
                minus_phase_termination_threshold=.01,                    \
                plus_phase_termination_condition = CONVERGENCE,           \
                plus_phase_termination_threshold=.01,                     \
                phase_convergence_function=Distance(metric=MAX_ABS_DIFF), \
                max_passes=None,                                          \
                additional_input_ports=None,                              \
                additional_output_ports=None)

    Subclass of `RecurrentTransferMechanism` that implements a single-layer auto-recurrent network using two-phases
    of execution and the `Contrastive Hebbian Learning algorithm
    <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_.  See `RecurrentTransferMechanism
    <RecurrentTransferMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    input_size : int : default 0
        specifies the size of the *INPUT* `InputPort <ContrastiveHebbian_Input>` and number of units in the
        `input_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`.

    hidden_size : int : default None
        specifies the number of units in the `hidden_field <ContrastiveHebbian_Fields>` of `current_activity
        <ContrastiveHebbianMechanism.current_activity>`.

    target_size : int : default None
        specifies the size of the *TARGET* `InputPort <ContrastiveHebbian_Input>`, number of units in the
        `input_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`,
        and of the `OUTPUT_ACTIVITY` `OutputPort <ContrastiveHebbian_Output>`.

    separated : bool : default True
        specifies whether `target_field <ContrastiveHebbian_Fields>` should be different from the `input_field
        <ContrastiveHebbian_Fields>`.

    mode : SIMPLE_HEBBIAN or None : default None
        specifies configuratoin that emulates standard `RecurrentTransferMechanism` using `Hebbian` learning
        (see `SIMPLE_HEBBIAN mode <ContrastiveHebbian_SIMPLE_HEBBIAN>` for details).

    combination_function : function : default None
        specifies function used to combine the *INPUT*, *RECURRENT* and *TARGET* (if specified) `InputPorts
        <ContrastiveHebbian_Input>`; must accept a 2d array with two or three items of the same length, and generate a
        result that is the same size as `recurrent_size <ContrastiveHebbianMechanism.recurrent_size>`;  if `None`,
        the ContrastiveHebbianMechanism's combination_function method is used.

    clamp : HARD_CLAMP or SOFT_CLAMP : default HARD_CLAMP
        specifies the manner in which the `value <InputPort.value>` of the  *INPUT* and *TARGET* (if specified)
        `InputPorts <ContrastiveHebbian_Input>` are combined with the *RECURRENT* `InputPort
        <ContrastiveHebbian_Input>` in each execution (see `clamp <ContrastiveHebbianMechanism.clamp>` for additional
        details.

    continuous : bool : default True
        specifies whether or not to reset `current_activity <ContrastiveHebbianMechanism.current_activity>`
        at the beginning of the `minus phase <ContrastiveHebbian_Minus_Phase>` of a trial.

    minus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        specifies the type of condition used to terminate the `minus_phase <ContrastiveHebbian_Minus_Phase>` of
        execution (see `minus_phase_termination_condition
        <ContrastiveHebbianMechanism.minus_phase_termination_condition>` for additional details).

    minus_phase_termination_threshold : float or int : default 0.01
        specifies the value used to determine when the `minus_phase <ContrastiveHebbian_Minus_Phase>` terminates;
        should be a float if `minus_phase_termination_condition
        <ContrastiveHebbianMechanism.minus_phase_termination_condition>` is *CONVERGENCE*, and int if it is *COUNT*
        (see `minus_phase_termination_threshold <ContrastiveHebbianMechanism.minus_phase_termination_threshold>`
        for additional details).

    plus_phase_termination_condition : COUNT or CONVERGENCE : default CONVERGENCE
        specifies the type of condition used to terminate the `plus_phase <ContrastiveHebbian_Plus_Phase>` of
        execution (see `plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`
        for additional details).

    plus_phase_termination_threshold : float or int : default 0.01
        specifies the value used to determine when the `minus_phase <ContrastiveHebbian_Plus_Phase>` terminates;
        should be a float if `plus_phase_termination_condition
        <ContrastiveHebbianMechanism.plus_phase_termination_condition>` is *CONVERGENCE*, and int if it is *COUNT*
        (see `plus_phase_termination_threshold <ContrastiveHebbianMechanism.plus_phase_termination_threshold>`
        for additional details).

    phase_convergence_function : function : default Distance(metric=MAX_ABS_DIFF)
        specifies the function that determines when a `phase of execution <ContrastiveHebbian_Execution>` is complete
        if the termination condition for that phase is specified as *CONVERGENCE*, by comparing `current_activity
        <ContrastiveHebbianMechanism.current_activity>` with the previous `value
        <Mechanism_Base._value>` of the Mechanism;  can be any function that takes two 1d arrays
        of the same length as `variable <Mechanism_Base.variable>` and returns a scalar value. The default
        is the `Distance` Function, using the `MAX_ABS_DIFF` metric  which computes the elementwise difference between
        two arrays and returns the difference with the maximum absolute value.

    max_passes : int : default 1000
        specifies maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the termination_threshold for that phase,
        after which an error occurs; if `None` is specified, execution may continue indefinitely or until an
        interpreter exception is generated.

    Attributes
    ----------

    input_size : int
        size of the *INPUT* `InputPort <ContrastiveHebbian_Input>` and `input_activity
        <ContrastiveHebbianMechanism.input_activity>`, and the number of units in the `input_field of
        <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`.

    hidden_size : int
        size of `hidden_activity <ContrastiveHebbianMechanism.input_activity>`, and number of units in
        `current_activity <ContrastiveHebbianMechanism.current_activity>` and the `hidden_field
        <ContrastiveHebbian_Fields>` of the *RECURRENT* `InputPort <ContrastiveHebbian_Input>`.

    target_size : int
        size of the *TARGET* `InputPort <ContrastiveHebbian_Input>` `if specified <ContrastiveHebbian_Creation>` and,
        if so, the number of units in `target_activity <ContrastiveHebbianMechanism.target_activity>`, the
        `target_field <ContrastiveHebbian_Fields>` of `current_activity <ContrastiveHebbianMechanism.current_activity>`,
        and the *OUTPUT_ACTIVITY* `OutputPort <ContrastiveHebbian_Output>`.

    target_start : int
        index of first unit of `target_field <ContrastiveHebbian_Fields>`.

    target_end : int
        index of first unit *after* `target_field <ContrastiveHebbian_Fields>`.

    separated : bool : default True
        determines whether `target_field <ContrastiveHebbian_Fields>` is different from `input_field
        <ContrastiveHebbian_Fields>` (`True`) or the same (`False`).

    recurrent_size : int
        size of *RECURRENT* `InputPort <ContrastiveHebbian_Input>`, `current_activity
        <ContrastiveHebbianMechanism.current_activity>`, and *CURRENT_ACTIVITY* `OutputPort
        <ContrastiveHebbian_Output>`.

    mode : SIMPLE_HEBBIAN or None
        indicates whether *SIMPLE_HEBBIAN* was used for configuration (see
        `SIMPLE_HEBBIAN mode <ContrastiveHebbian_SIMPLE_HEBBIAN>` for details).

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the *CURRENT_ACTIVITY* `OutputPort
        <ContrastiveHebbian_Output>` to the *RECURRENT* `InputPort <ContrastiveHebbian_Input>`.

    combination_function : method or function
        used to combine `value <InputPort.value>` of the *INPUT* and *TARGET* (if specified)  `InputPorts
        <ContrastiveHebbian_Input>` with that of the *RECURRENT* `InputPort <ContrastiveHebbian_Input>` to determine
        the `variable <CurrentHebbianMechanism.variable>` passed to the Mechanism's `integrator_function
        <TransferMechanism.integrator_function>` and/or its `function <Mechanism_Base.function>`
        (see `ContrastiveHebbian_Execution` for details).

    clamp : HARD_CLAMP or SOFT_CLAMP
        determines whether the `value <InputPort.value>` of the *INPUT* and *TARGET* (if specified) `InputPorts
        <ContrastiveHebbian_Input>` replace (*HARD_CLAMP*) or are added to (*SOFT_CLAMP*) the `value <InputPort.value>`
        of the *RECURRENT* InputPort by `combination_function <ContrastiveHebbianMechanism.combination_function>`.

    continuous : bool : default True
        determines whether or not `current_activity <ContrastiveHebbianMechanism.current_activity>` is reset
        at the beginning of the `minus phase <ContrastiveHebbian_Minus_Phase>` of execution. If `False`, it is set to `initial_value
        <Mechanism_Base.initial_value>`.

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
        <ContrastiveHebbianMechanism.current_activity>` if *TARGET* `InputPort is specified
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

    delta : scalar
        value returned by `phase_convergence_function <RecurrentTransferMechanism.phase_convergence_function>`;
        used to determined when `is_converged <RecurrentTransferMechanism.is_converged>` is `True`.

    is_converged : bool
        indicates when a `phase of execution <ContrastiveHebbian_Execution>` is complete, if the termination condition
        for that phase is specified as *CONVERGENCE*.  `True` when `delta <ContrastiveHebbianMechanism.delta>` is less
        than or equal to the termination_threshold speified for the corresponding phase.

    phase_convergence_function : function
        determines when a `phase of execution <ContrastiveHebbian_Execution>` is complete if the termination
        condition for that phase is specified as *CONVERGENCE*.  Compares the value of `current_activity
        <ContrastiveHebbianMechanism.current_activity>` with the
        previous `value <Mechanism_Base.value>`; result is
        assigned as the value of `delta
        <ContrastiveHebbianMechanism.delta>.

    minus_phase_termination_condition : CONVERGENCE or COUNT: default CONVERGENCE
        determines the type of condition used to terminate the `minus_phase <ContrastiveHebbian_Minus_Phase>` of
        execution.  If it is *CONVERGENCE*, execution continues until the value returned by `phase_convergence_function
        <ContrastiveHebbianMechanism.phase_convergence_function>` is less than or equal to
        `minus_phase_termination_threshold <ContrastiveHebbianMechanism.minus_phase_termination_threshold>`;
        if it is *COUNT*, the Mechanism is executed the number of times specified in
        `minus_phase_termination_threshold <ContrastiveHebbianMechanism.minus_phase_termination_threshold>`.

    minus_phase_termination_threshold : float or int
        the value used for the specified `minus_phase_termination_condition
        <ContrastiveHebbianMechanism.minus_phase_termination_condition>` to determine when the
        `minus phase of execution <ContrastiveHebbian_Minus_Phase>` terminates.

    plus_phase_termination_condition : CONVERGENCE or COUNT : default CONVERGENCE
        determines the type of condition used to terminate the `plus_phase <ContrastiveHebbian_Plus_Phase>` of
        execution.  If it is *CONVERGENCE*, execution continues until the value returned by `phase_convergence_function
        <ContrastiveHebbianMechanism.phase_convergence_function>` is less than or equal to
        `plus_phase_termination_threshold <ContrastiveHebbianMechanism.plus_phase_termination_threshold>`;
        if it is *COUNT*, the Mechanism is executed the number of times specified in
        `plus_phase_termination_threshold <ContrastiveHebbianMechanism.plus_phase_termination_threshold>`.

    plus_phase_termination_threshold : float or int
        the value used for the specified `plus_phase_termination_condition
        <ContrastiveHebbianMechanism.plus_phase_termination_condition>` to determine when the `plus phase of execution
        <ContrastiveHebbian_Plus_Phase>` terminates.

    max_passes : int or None
        determines the maximum number of executions (`passes <TimeScale.PASS>`) that can occur in an `execution phase
        <ContrastiveHebbian_Execution>` before reaching the corresponding termination_threshold for that phase,
        after which an error occurs; if `None` is specified, execution may continue indefinitely or until an
        interpreter exception is generated.

    execution_phase : bool
        indicates current `phase of execution <ContrastiveHebbian_Execution>`.
        `False` = `minus phase <ContrastiveHebbian_Minus_Phase>`;
        `True` = `plus phase <ContrastiveHebbian_Plus_Phase>`.

    output_ports : ContentAddressableList[OutputPort]
        contains the following `OutputPorts <OutputPort>` by default:

        * *OUTPUT_ACTIVITY*, the `primary OutputPort  <OutputPort_Primary>` of the Mechanism, the `value
          <OutputPort.value>` of which is `target_activity <ContrastiveHebbianMechanism.target_activity>` if a
          *TARGET* `InputPort <ContrastiveHebbian_Input>` is implemented;  otherwise,  `input_activity
          <ContrastiveHebbianMechanism.input_activity>` is assigned as its `value <OutputPort.value>`.

        * *CURRENT_ACTIVITY* -- the `value <OutputPort.value>` of which is a 1d array containing the activity
          of the ContrastiveHebbianMechanism after each execution;  at the end of an `execution sequence
          <ContrastiveHebbian_Execution>`, it is assigned the value of `plus_phase_activity
          <ContrastiveHebbianMechanism.plus_phase_activity>`.

        * *ACTIVITY_DIFFERENCE*, the `value <OutputPort.value>` of which is a 1d array with the elementwise
          differences in activity between the plus and minus phases at the end of an `execution sequence
          <ContrastiveHebbian_Execution>`.

    standard_output_ports : list[str]
        list of `Standard OutputPorts <OutputPort_Standard>` that includes the following in addition to the
        `standard_output_ports <RecurrentTransferMechanism.standard_output_ports>` of a `RecurrentTransferMechanism`:

        .. _OUTPUT_ACTIVITY:

        *OUTPUT_ACTIVITY* : 1d np.array
            array with activity of the `target_field <ContrastiveHebbian_Fields>` of `current_activity
            <ContrastiveHebbianMechanism.current_activity>` if a *TARGET* `InputPort is specified
            <ContrastiveHebbian_Input>`;  otherwise, has activity of the `input_field <ContrastiveHebbian_Fields>` of
            `current_activity <ContrastiveHebbianMechanism.current_activity>`.

        .. _CURRENT_ACTIVITY:

        *CURRENT_ACTIVITY* : 1d np.array
            array with `current_activity <ContrastiveHebbianMechanism.current_activity>`.

        .. _ACTIVITY_DIFFERENCE:

        *ACTIVITY_DIFFERENCE* : 1d np.array
            array of element-wise differences between `plus_phase_activity
            <ContrastiveHebbianMechanism.plus_phase_activity>` and `minus_phase_activity
            <ContrastiveHebbianMechanism.minus_phase_activity>`.

        .. _MINUS_PHASE_ACTIVITY:

        *MINUS_PHASE_ACTIVITY* : 1d np.array
            array `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`
            (i.e., activity at the end of the `minus phase of execution <ContrastiveHebbian_Minus_Phase>`.

        .. _PLUS_PHASE_ACTIVITY:

        *PLUS_PHASE_ACTIVITY* : 1d np.array
            array `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>`
            (i.e., activity at the end of the `plus phase of execution <ContrastiveHebbian_Plus_Phase>`.


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
                    :type: ``numpy.ndarray``

                clamp
                    see `clamp <ContrastiveHebbianMechanism.clamp>`

                    :default value: `HARD_CLAMP`
                    :type: ``str``

                combination_function
                    see `combination_function <ContrastiveHebbianMechanism.combination_function>`

                    :default value: None
                    :type:

                continuous
                    see `continuous <ContrastiveHebbianMechanism.continuous>`

                    :default value: True
                    :type: ``bool``

                current_activity
                    see `current_activity <ContrastiveHebbianMechanism.current_activity>`

                    :default value: None
                    :type:

                current_termination_condition
                    see `current_termination_condition <ContrastiveHebbianMechanism.current_termination_condition>`

                    :default value: None
                    :type:

                current_termination_threshold
                    see `current_termination_threshold <ContrastiveHebbianMechanism.current_termination_threshold>`

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

                    :default value: 0
                    :type: ``int``

                input_activity
                    see `input_activity <ContrastiveHebbianMechanism.input_activity>`

                    :default value: None
                    :type:
                    :read only: True

                input_size
                    see `input_size <ContrastiveHebbianMechanism.input_size>`

                    :default value: None
                    :type:

                learning_function
                    see `learning_function <ContrastiveHebbianMechanism.learning_function>`

                    :default value: `ContrastiveHebbian`
                    :type: `Function`

                max_passes
                    see `max_passes <ContrastiveHebbianMechanism.max_passes>`

                    :default value: 1000
                    :type: ``int``

                minus_phase_activity
                    see `minus_phase_activity <ContrastiveHebbianMechanism.minus_phase_activity>`

                    :default value: None
                    :type:

                minus_phase_termination_condition
                    see `minus_phase_termination_condition <ContrastiveHebbianMechanism.minus_phase_termination_condition>`

                    :default value: `CONVERGENCE`
                    :type: ``str``

                minus_phase_termination_threshold
                    see `minus_phase_termination_threshold <ContrastiveHebbianMechanism.minus_phase_termination_threshold>`

                    :default value: 0.01
                    :type: ``float``

                mode
                    see `mode <ContrastiveHebbianMechanism.mode>`

                    :default value: None
                    :type:

                output_activity
                    see `output_activity <ContrastiveHebbianMechanism.output_activity>`

                    :default value: None
                    :type:
                    :read only: True

                phase_convergence_function
                    see `phase_convergence_function <ContrastiveHebbianMechanism.phase_convergence_function>`

                    :default value: `Distance`
                    :type: `Function`

                phase_convergence_threshold
                    internal parameter, used by is_converged;  assigned the value of the termination_threshold for
                    the current `phase of execution <ContrastiveHebbian_Execution>`.

                    :default value: 0.01
                    :type: ``float``

                phase_execution_count
                    see `phase_execution_count <ContrastiveHebbianMechanism.phase_execution_count>`

                    :default value: 0
                    :type: ``int``

                phase_terminated
                    see `phase_terminated <ContrastiveHebbianMechanism.phase_terminated>`

                    :default value: False
                    :type: ``bool``

                plus_phase_activity
                    see `plus_phase_activity <ContrastiveHebbianMechanism.plus_phase_activity>`

                    :default value: None
                    :type:

                plus_phase_termination_condition
                    see `plus_phase_termination_condition <ContrastiveHebbianMechanism.plus_phase_termination_condition>`

                    :default value: `CONVERGENCE`
                    :type: ``str``

                plus_phase_termination_threshold
                    see `plus_phase_termination_threshold <ContrastiveHebbianMechanism.plus_phase_termination_threshold>`

                    :default value: 0.01
                    :type: ``float``

                separated
                    see `separated <ContrastiveHebbianMechanism.separated>`

                    :default value: True
                    :type: ``bool``

                target_activity
                    see `target_activity <ContrastiveHebbianMechanism.target_activity>`

                    :default value: None
                    :type:
                    :read only: True

                target_size
                    see `target_size <ContrastiveHebbianMechanism.target_size>`

                    :default value: 0
                    :type: ``int``
        """
        variable = np.array([[0, 0]])
        current_activity = Parameter(None, aliases=['recurrent_activity'])
        plus_phase_activity = None
        minus_phase_activity = None
        current_termination_condition = None
        current_termination_threshold = None
        phase_execution_count = 0
        phase_terminated = False

        input_size = Parameter(None, stateful=False, loggable=False)
        hidden_size = Parameter(0, stateful=False, loggable=False)
        target_size = Parameter(0, stateful=False, loggable=False)
        separated = Parameter(True, stateful=False, loggable=False)
        mode = Parameter(None, stateful=False, loggable=False)
        continuous = Parameter(True, stateful=False, loggable=False)
        clamp = Parameter(HARD_CLAMP, stateful=False, loggable=False)
        combination_function = Parameter(None, stateful=False, loggable=False)
        phase_convergence_function = Parameter(Distance(metric=MAX_ABS_DIFF), stateful=False, pnl_internal=True, loggable=False)
        phase_convergence_threshold = Parameter(0.01, modulable=True, pnl_internal=True, loggable=False)

        minus_phase_termination_condition = Parameter(CONVERGENCE, stateful=False, loggable=False)
        plus_phase_termination_condition = Parameter(CONVERGENCE, stateful=False, loggable=False)
        learning_function = Parameter(
            ContrastiveHebbian,
            stateful=False,
            loggable=False,
            reference=True
        )
        max_passes = Parameter(1000, stateful=False)

        output_activity = Parameter(None, read_only=True, getter=_CHM_output_activity_getter)
        input_activity = Parameter(None, read_only=True, getter=_CHM_input_activity_getter)
        hidden_activity = Parameter(None, read_only=True, getter=_CHM_hidden_activity_getter)
        target_activity = Parameter(None, read_only=True, getter=_CHM_target_activity_getter)

        execution_phase = Parameter(None, read_only=True)
        # is_finished_ = Parameter(False, read_only=True)

        minus_phase_termination_threshold = Parameter(0.01, modulable=True)
        plus_phase_termination_threshold = Parameter(0.01, modulable=True)


    standard_output_ports = RecurrentTransferMechanism.standard_output_ports.copy()
    standard_output_ports.extend([{NAME:OUTPUT_ACTIVITY,
                                    VARIABLE:OUTPUT_ACTIVITY_ATTR},
                                   {NAME:CURRENT_ACTIVITY,
                                    VARIABLE:CURRENT_ACTIVITY_ATTR},
                                   {NAME:ACTIVITY_DIFFERENCE,
                                    VARIABLE:[PLUS_PHASE_ACTIVITY_ATTR, MINUS_PHASE_ACTIVITY_ATTR],
                                    FUNCTION: lambda v: v[0] - v[1]},
                                   {NAME:MINUS_PHASE_ACTIVITY,
                                    VARIABLE:MINUS_PHASE_ACTIVITY_ATTR},
                                   {NAME:PLUS_PHASE_ACTIVITY,
                                    VARIABLE:PLUS_PHASE_ACTIVITY_ATTR},
                                   ])
    standard_output_port_names = RecurrentTransferMechanism.standard_output_port_names.copy()
    standard_output_port_names = [i['name'] for i in standard_output_ports]

    @tc.typecheck
    def __init__(self,
                 input_size:int,
                 hidden_size:tc.optional(int)=None,
                 target_size:tc.optional(int)=None,
                 separated: tc.optional(bool) = None,
                 mode:tc.optional(tc.enum(SIMPLE_HEBBIAN))=None,
                 continuous: tc.optional(bool) = None,
                 clamp:tc.optional(tc.enum(SOFT_CLAMP, HARD_CLAMP))=None,
                 combination_function:tc.optional(is_function_type)=None,
                 function=None,
                 matrix=None,
                 auto=None,
                 hetero=None,
                 integrator_function=None,
                 initial_value=None,
                 noise=None,
                 integration_rate: is_numeric_or_none=None,
                 integrator_mode: tc.optional(bool) = None,
                 clip=None,
                 minus_phase_termination_condition:tc.optional(tc.enum(CONVERGENCE, COUNT))=None,
                 minus_phase_termination_threshold: tc.optional(float) = None,
                 plus_phase_termination_condition:tc.optional(tc.enum(CONVERGENCE, COUNT))=None,
                 plus_phase_termination_threshold: tc.optional(float) = None,
                 phase_convergence_function: tc.optional(tc.any(is_function_type)) = None,
                 max_passes:tc.optional(int)=None,
                 enable_learning: tc.optional(bool) = None,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.optional(tc.any(is_function_type)) = None,
                 additional_input_ports:tc.optional(tc.optional(tc.any(list, dict))) = None,
                 additional_output_ports:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None,
                 **kwargs):
        """Instantiate ContrastiveHebbianMechanism
        """

        if mode == SIMPLE_HEBBIAN:
            hidden_size=0
            # target_size=0
            separated = False
            clamp = SOFT_CLAMP
            continuous = False
            learning_function = Hebbian

        self.recurrent_size = input_size + (hidden_size or 0)
        if separated and target_size:
            self.recurrent_size += target_size
            self.target_start = input_size + (hidden_size or 0)
            self._target_included = True
        else:
            self.target_start = 0
            self._target_included = False
        self.target_end = self.target_start + target_size
        size = self.recurrent_size

        default_variable = [np.zeros(input_size), np.zeros(self.recurrent_size)]
        # Set InputPort sizes in _instantiate_input_ports,
        #    so that there is no conflict with parsing of Mechanism's size
        input_ports = [INPUT, RECURRENT]
        if self._target_included:
            default_variable.append(np.zeros(target_size))
            input_ports.append(TARGET)

        if additional_input_ports:
            if isinstance(additional_input_ports, list):
                input_ports += additional_input_ports
            else:
                input_ports.append(additional_input_ports)

        combination_function = combination_function or self.combination_function

        output_ports = [OUTPUT_ACTIVITY, CURRENT_ACTIVITY, ACTIVITY_DIFFERENCE]
        if additional_output_ports:
            if isinstance(additional_output_ports, list):
                output_ports += additional_output_ports
            else:
                output_ports.append(additional_output_ports)

        super().__init__(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports,
            combination_function=combination_function,
            function=function,
            matrix=matrix,
            auto=auto,
            hetero=hetero,
            has_recurrent_input_port=True,
            integrator_function=integrator_function,
            initial_value=initial_value,
            noise=noise,
            integrator_mode=integrator_mode,
            integration_rate=integration_rate,
            clip=clip,
            enable_learning=enable_learning,
            learning_rate=learning_rate,
            learning_function=learning_function,
            learning_condition=CONVERGENCE,
            output_ports=output_ports,
            mode=mode,
            minus_phase_termination_condition=minus_phase_termination_condition,
            minus_phase_termination_threshold=minus_phase_termination_threshold,
            plus_phase_termination_condition=plus_phase_termination_condition,
            plus_phase_termination_threshold=plus_phase_termination_threshold,
            phase_convergence_function=phase_convergence_function,
            phase_convergence_threshold=minus_phase_termination_threshold,
            max_passes=max_passes,
            continuous=continuous,
            input_size=input_size,
            hidden_size=hidden_size,
            target_size=target_size,
            separated=separated,
            clamp=clamp,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Make sure that the size of the INPUT and TARGET InputPorts are <= size of RECURRENT InputPort
        if self._target_included and self.input_size != self.target_size:
            raise ContrastiveHebbianError("{} is {} for {} must equal {} ({}) must equal {} ({})} ".
                                          format(repr(SEPARATED), repr(True), self.name,
                                                 repr(INPUT_SIZE), self.input_size,
                                                 repr(TARGET_SIZE), self.target_size))

    def _instantiate_input_ports(self, input_ports=None, reference_value=None, context=None):

        # Assign InputPort specification dictionaries for required InputPorts
        sizes = dict(INPUT=self.input_size, RECURRENT=self.recurrent_size, TARGET=self.target_size)
        for i, input_port in enumerate((s for s in self.input_ports if s in {INPUT, TARGET, RECURRENT})):
            self.input_ports[i] = {NAME:input_port, SIZE: sizes[input_port]}

        super()._instantiate_input_ports(input_ports, reference_value, context)

        self.input_ports[RECURRENT].internal_only = True
        if self._target_included:
            self.input_ports[TARGET].internal_only = True

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function=function, context=context)

        # Set minus_phase activity, plus_phase, current_activity and initial_value
        #    all  to zeros with size of Mechanism's array
        # Should be OK to use attributes here because initialization should only occur during None context
        self._set_multiple_parameter_values(
            context,
            initial_value=self.input_ports[RECURRENT].socket_template,
            current_activity=self.input_ports[RECURRENT].socket_template,
            minus_phase_activity=self.input_ports[RECURRENT].socket_template,
            plus_phase_activity=self.input_ports[RECURRENT].socket_template,
            execution_phase=None,
        )
        self.defaults.initial_value = copy.deepcopy(self.input_ports[RECURRENT].socket_template)
        self.defaults.current_activity = copy.deepcopy(self.input_ports[RECURRENT].socket_template)
        self.defaults.minus_phase_activity = copy.deepcopy(self.input_ports[RECURRENT].socket_template)
        self.defaults.plus_phase_activity = copy.deepcopy(self.input_ports[RECURRENT].socket_template)
        self.defaults.execution_phase = None

        if self._target_included:
            self.parameters.output_activity._set(self.input_ports[TARGET].socket_template, context)

    @tc.typecheck
    def _instantiate_recurrent_projection(self,
                                          mech: Mechanism,
                                          # this typecheck was failing, I didn't want to fix (7/19/17 CW)
                                          # matrix:is_matrix=HOLLOW_MATRIX,
                                          matrix=HOLLOW_MATRIX,
                                          context=None):
        """Instantiate an AutoAssociativeProjection from Mechanism to itself
        """
        if isinstance(matrix, str):
            size = len(mech.defaults.variable[0])
            matrix = get_matrix(matrix, size, size)

        return AutoAssociativeProjection(owner=mech,
                                         sender=self.output_ports[CURRENT_ACTIVITY],
                                         receiver=self.input_ports[RECURRENT],
                                         matrix=matrix,
                                         name=mech.name + ' recurrent projection')

    def _execute(self,
                 variable=None,
                 context=None,
                 function_variable=None,
                 runtime_params=None,
                 ):

        # Initialize execution_phase as minus_phase
        if self.parameters.execution_phase._get(context) is None:
            self.parameters.execution_phase._set(MINUS_PHASE, context)

        if self.parameters.execution_phase._get(context) == MINUS_PHASE:
            self.parameters.current_termination_threshold._set(
                    self.parameters.minus_phase_termination_threshold._get(context), context)
            self.parameters.current_termination_condition._set(self.minus_phase_termination_condition, context)
            self.parameters.phase_execution_count._set(0, context)

        if self.parameters.is_finished_flag._get(context):
        # if self.parameters.is_finished_._get(context):
            # If current execution follows completion of a previous trial,
            #    zero activity for input from recurrent projection so that
            #    input does not contain residual activity of previous trial
            variable[RECURRENT_INDEX] = self.input_ports[RECURRENT].socket_template

        # self.parameters.is_finished_._set(False, context)
        self.parameters.is_finished_flag._set(False, context)

        # Note _parse_function_variable selects actual input to function based on execution_phase
        current_activity = super()._execute(variable,
                                            context=context,
                                            runtime_params=runtime_params,
                                            )

        self.parameters.phase_execution_count._set(self.parameters.phase_execution_count._get(context) + 1, context)

        current_activity = np.squeeze(current_activity)
        # Set value of primary OutputPort to current activity
        self.parameters.current_activity._set(current_activity, context)

        # This is the first trial, so can't test for convergence
        #    (since that requires comparison with value from previous trial)
        # (here, the "previous value" is the current parameter value
        # because it has not been updated with the results of this
        # execution yet)
        if self.parameters.value._get(context) is None:
            return current_activity

        current_termination_condition = self.parameters.current_termination_condition._get(context)

        if current_termination_condition == CONVERGENCE:
            self.parameters.phase_convergence_threshold._set(
                    self.parameters.current_termination_threshold._get(context),context)
            self.parameters.phase_terminated._set(
                    self.is_converged(np.atleast_2d(current_activity), context), context)
        elif current_termination_condition == COUNT:
            self.parameters.phase_terminated._set(
                (self.parameters.phase_execution_count._get(context) ==
                 self.parameters.current_termination_threshold._get(context)),
                context)
        else:
            phase_str = repr('PLUS_PHASE') if self.parameters.execution_phase._get(context) == PLUS_PHASE \
                                           else repr('MINUS_PHASE')
            raise ContrastiveHebbianError(f"Unrecognized {repr('current_termination_condition')} specification "
                                          f"({current_termination_condition}) in execution of {phase_str} of "
                                          f"{self.name}.")

        if self.parameters.phase_terminated._get(context):
            # Terminate if this is the end of the plus phase, prepare for next trial
            if self.parameters.execution_phase._get(context) == PLUS_PHASE:
                # Store activity from last execution in plus phase
                self.parameters.plus_phase_activity._set(current_activity, context)
                # # Set value of primary outputPort to activity at end of plus phase
                self.parameters.current_activity._set(current_activity, context)
                self.parameters.is_finished_flag._set(True, context)

            # Otherwise, prepare for start of plus phase on next execution
            else:
                # Store activity from last execution in plus phase
                self.parameters.minus_phase_activity._set(self.parameters.current_activity._get(context), context)
                # Use initial_value attribute to initialize, for the minus phase,
                #    both the integrator_function's previous_value
                #    and the Mechanism's current activity (which is returned as its input)
                if not self.continuous and self.parameters.integrator_mode._get(context):
                    self.reset(self.initial_value, context=context)
                    self.parameters.current_activity._set(self.parameters.initial_value._get(context), context)
                self.parameters.current_termination_threshold._set(self.plus_phase_termination_threshold, context)
                self.parameters.current_termination_condition._set(self.plus_phase_termination_condition, context)

            # Switch execution_phase
            self.parameters.execution_phase._set(not self.parameters.execution_phase._get(context), context)
            self.parameters.phase_execution_count._set(0, context)

        return current_activity
        # return self.current_activity

    def _parse_function_variable(self, variable, context=None):
        function_variable = self.combination_function(variable=variable, context=context)
        return super(RecurrentTransferMechanism, self)._parse_function_variable(function_variable, context=context)

    def _parse_phase_convergence_function_variable(self, variable):
        # determines shape only
        return np.asarray([variable[0], variable[0]])

    def combination_function(self, variable=None, context=None):
        # IMPLEMENTATION NOTE: use try and except here for efficiency: care more about execution than initialization
        # IMPLEMENTATION NOTE: separated vs. overlapping input and target handled by assignment of target_start in init

        MINUS_PHASE_INDEX = INPUT_INDEX
        PLUS_PHASE_INDEX = TARGET_INDEX

        try:  # Execution
            if self.parameters.execution_phase._get(context) == PLUS_PHASE:
                if self.clamp == HARD_CLAMP:
                    variable[RECURRENT_INDEX][:self.input_size] = variable[MINUS_PHASE_INDEX]
                    if self.mode == SIMPLE_HEBBIAN:
                        return variable[RECURRENT_INDEX]
                    else:
                        variable[RECURRENT_INDEX][self.target_start:self.target_end] = variable[PLUS_PHASE_INDEX]
                else:
                    variable[RECURRENT_INDEX][:self.input_size] += variable[MINUS_PHASE_INDEX]
                    if self.mode == SIMPLE_HEBBIAN:
                        return variable[RECURRENT_INDEX]
                    else:
                        variable[RECURRENT_INDEX][self.target_start:self.target_end] += variable[PLUS_PHASE_INDEX]
            else:
                if self.mode == SIMPLE_HEBBIAN:
                    return variable[RECURRENT_INDEX]
                if self.clamp == HARD_CLAMP:
                    variable[RECURRENT_INDEX][:self.input_size] = variable[MINUS_PHASE_INDEX]
                else:
                    variable[RECURRENT_INDEX][:self.input_size] += variable[MINUS_PHASE_INDEX]
        except:  # Initialization
            pass

        return variable[RECURRENT_INDEX]

    def delta(self, value=NotImplemented, context=None):
        if value is NotImplemented:
            self.phase_convergence_function([
                self.parameters.value._get(context)[0],
                self.parameters.value.get_previous(context)[0]
            ])
        else:
            return self.phase_convergence_function([value[0], self.parameters.value._get(context)[0]])

    @handle_external_context()
    def is_converged(self, value=NotImplemented, context=None):
        # Check for convergence
        if (
            self.phase_convergence_threshold is not None
            and self.parameters.value.get_previous(context) is not None
            and self.initialization_status != ContextFlags.INITIALIZING
        ):
            if self.delta(value, context) <= self.phase_convergence_threshold:
                return True
            elif self.get_current_execution_time(context).pass_ >= self.max_passes:
                phase_str = repr('PLUS_PHASE') if self.parameters.execution_phase._get(context) == PLUS_PHASE \
                    else repr('MINUS_PHASE')
                raise ContrastiveHebbianError(f"Maximum number of executions ({self.max_passes}) has occurred "
                                              f"before reaching convergence_threshold "
                                              f"({self.phase_convergence_threshold}) for {self.name} in "
                                              f"{phase_str} of trial {self.get_current_execution_time(context).trial} "
                                              f"of run {self.get_current_execution_time(context).run}.")
            else:
                return False
        # Otherwise just return True
        else:
            return None

    @property
    def _learning_signal_source(self):
        """Override default to use ACTIVITY_DIFFERENCE as source of learning signal
        """
        return self.output_ports[ACTIVITY_DIFFERENCE]

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

    @property
    def recurrent_size(self):
        return self._recurrent_size

    @recurrent_size.setter
    def recurrent_size(self, value):
        self._recurrent_size = value
