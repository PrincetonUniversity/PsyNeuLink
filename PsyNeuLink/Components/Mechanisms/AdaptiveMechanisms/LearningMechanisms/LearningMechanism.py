# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningMechanism *******************************************************

"""
.. _LearningMechanism_Overview:

Overview
--------

A LearningMechanism is an `AdaptiveMechanism <AdaptiveMechanism>` that modifies the `matrix <MappingProjection.matrix>`
of a `MappingProjection`.  Its function takes an `error_signal` (usually the output of an `ObjectiveMechanism` or
another `LearningMechanism`) and generates a `learning_signal` that is used to modify the MappingProjection by way of
a `LearningProjection`.  The MappingProjection(s) modified by a LearningMechanism must project from one
`ProcessingMechanism` to another in same `System` or `Process`. The learning components of a System can be displayed
using the System's `show_graph` method with its **show_learning** argument assigned `True`. LearningMechanisms
are executed after all of the ProcessingMechanisms in a System or Process have been executed, and before any
`ControlMechanisms <ControlMechanism>` of the System have been executed (see `System Execution <System_Execution>`).

.. _LearningMechanism_Creation:

Creating a LearningMechanism
----------------------------

LearningMechanisms can be created in any of the ways used to `create Mechanisms <Mechanism_Creation>`.
More commonly, however, they are created automatically when:

* the learning attribute is specified for a `System <System_Execution_Learning>` or `Process <Process_Learning>`;
..
* a `LearningProjection` (or the keyword *LEARNING*) is specified as the second item of a
  `tuple used to specify the matrix parameter <Mapping_Matrix_Specification>` of a `MappingProjection` in
  the `pathway <Process_Base.pathway>` of a `Process`.
..
* a `LearningProjection` is created without specifying its `sender <LearningProjection.sender>` attribute.

In these instances, an `ObjectiveMechanism`, `LearningProjection <LearningProjection>`, and any additional Projections
required to implement learning that do not already exist are also instantiated.  This is described below, under
`Learning Configurations <LearningMechanism_Learning_Configurations>`.


.. _LearningMechanism_Structure:

Structure
---------

A LearningMechanism has three `InputStates <InputState>`, a learning `function <LearningMechanism.function>`,
and two types of `OutputStates <OutputState>` that are used, respectively, to receive, compute, and transmit the
information needed to modify the MappingProjection(s) for which it is responsible.  In addition, it has several
attributes that govern its operation.  These are all described below.

.. _LearningMechanism_InputStates:

InputStates
~~~~~~~~~~~

These receive the information required by the LearningMechanism's `function <LearningMechanism.function>`.  They are
listed in the LearningMechanism's `input_states <LearningMechanism.input_states>` attribute, and have the following
names and roles (shown in the `figure <LearningMechanism_Single_Layer_Learning_Figure>` below):

.. _LearningMechanism_Activation_Input:

* *ACTIVATION_INPUT* - receives the value of the input to the MappingProjection being learned (that is, the
  `value <MappingProjection.value>` of the MappingProjection's `sender <MappingProjection.sender>`).
  The value is assigned as the first item of the LearningMechanism's `variable <LearningMechanism.variable>` attribute.

.. _LearningMechanism_Activation_Output:

* *ACTIVATION_OUTPUT* - receives the value of the LearningMechanism's
  `error_source <LearningMechanism_Additional_Attributes>` (that is, the output of the *ProcessingMechanism* to which
  the MappingProjection being learned projects).  By default, the `primary OutputState <OutputState_Primary>` of the
  `error_source` is used.  However, a different OutputState can be designated in the constructor for the `error_source`,
  by assigning a `parameter specification dictionary <ParameterState_Specification>` to the **params** argument of its
  constructor, with an entry that uses *MONITOR_FOR_LEARNING* as its key and a list containing the desired
  OutputState(s) as its value. The `value <InputState.value>` of the *ACTIVATION_OUTPUT* InputState is assigned as
  the second item of the LearningMechanism's `variable <LearningMechanism.variable>` attribute.

.. _LearningMechanism_Input_Error_Signal:

* *ERROR_SIGNAL* - receives the value of an `error_signal <LearningMechanism.error_signal>` from either a
  `ComparatorMechanism` or another LearningMechanism. If the MappingProjection being learned projects to the `TERMINAL`
  Mechanism of the Process or System being learned, or is not part of a
  `multilayer learning sequence <LearningMechanism_Multilayer_Learning>`, then the `error_signal` comes from a
  ComparatorMechanism. If the MappingProjection being learned is part of a `multilayer learning sequence
  <_LearningMechanism_Multilayer_Learning>`, then the `error_signal` comes from the next LearningMechanism in the sequence
  (i.e., the one associated with the `error_source`).  The `value <InputState.value>` of the *ERROR_SIGNAL* InputState
  is assigned as the third item of the LearningMechanism's `variable <LearningMechanism.variable>` attribute.

   .. note::
      The value of a LearningMechanism's *ERROR_SIGNAL* InputState is distinct from its
      `error_signal <LearningMechanism.error_signal>` attribute. The former is received from a `ComparatorMechanism`
      or another LearningMechanism, whereas the latter is generated as a result of the LearningMechanism's
      `function <LearningMechanism.function>` (and potentially passed on to other LearningMechanisms).

.. _LearningMechanism_Function:

Learning Function
~~~~~~~~~~~~~~~~~

The `function <LearningMechanism.function>` of a LearningMechanism uses the three values received by the Mechanism's
InputStates (described `above <LearningMechanism_InputStates>` to calculate the value of its `learning_signal
<LearningMechanism.learning_signal>` and `error_signal <LearningMechanism.error_signal>` attributes.  The
`learning_signal` is the set of changes to the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection`
being learned, calculated to reduce the value of the LearningMechanism's *ERROR_SIGNAL* InputState .  In `multilayer
learning <LearningMechanism_Multilayer_Learning>`, the `error_signal` reflects the contribution made to value of the
*ERROR_SIGNAL* InputState by the input to the MappingProjection being learned, weighted by the current value of its
`matrix <MappingProjection.matrix>` parameter (i.e., before it has been modified). The default `function
<LearningMechanism.function>` of a LearningMechanism is `BackPropagation` (also known as the *Generalized Delta Rule*;
see `Rumelhart et al., 1986 <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_).  However, it can be
assigned any other PsyNeuLink `LearningFunction`, or any other Python function that takes as its input a value with
three 1d arrays or lists, and returns two 1d arrays or lists.  The two values it returns are assigned to the
LearningMechanism's `learning_signal <LearningMechanism.learning_signal>` and `error_signal
<LearningSignal.error_signal>` attributes, respectively, as well as to its two OutputStates, as described below.

.. _LearningMechanism_OutputStates:

OutputStates
~~~~~~~~~~~~

By default, a LearningMechanism has two OutputStates that receive, respectively, the values of the
`error_signal <LearningMechanism.error_signal>` and  `learning_signal <LearningMechanism.learning_signal>` attributes
(returned by the LearningMechanism's `function <LearningMechanism.function>`).

.. _LearningMechanism_Output_Error_Signal:

* *ERROR_SIGNAL* - this receives the value of the `error_signal <LearningMechanism.error_signal>` used to calculate
  the `learning_signal <LearningMechanism.learning_signal>`, which is the `value <InputState.value>` of the
  LearningMechanism's *ERROR_SIGNAL* InputState (see `above <LearningMechanism_Input_Error_Signal>`) possibly weighted
  by the contribution of the MappingProjection being learned and the output of the `error_source`.  This is always a
  LearningMechanism's first (i.e., `primary <OutputState_Primary>`) OutputState, and is always named *ERROR_SIGNAL*.
  Its value is assigned as the first item of the LearningMechanism's `output_values <LearningMechanism.output_values>`
  attribute.  If the LearningMechanism is part of a `multilayer learning sequence
  <LearningMechanism_Multilayer_Learning>`, the *ERROR_SIGNAL* OutputState is assigned a Projection to the
  LearningMechanism for the previous MappingProjection being learned in the sequence - see
  `figure <LearningMechanism_Multilayer_Learning_Figure>` below).

.. _LearningMechanism_Learning_Signal:

* `LearningSignals <LearningSignal>` - these are a special type of OutputState, that receive the matrix of weight
  changes calculated by a LearningMechanism's `function <LearningMechanism.function>`, and use this to modify the
  `matrix <MappingProjection.matrix>` parameter of the `MappingProjection(s) <MappingProjection>` being learned.
  By default, a LearningMechanism has just one LearningSignal.  It is assigned as the second item in the list of the
  LearningMechanism's OutputStates (i.e., in its `output_states <LearningMechanism.output_states>` attribute).  Its
  `value <LearningSignal.value>` is assigned as the value of the LearningMechanism's `learning_signal` attribute, and
  as the second item of the LearningMechanism's `output_values <LearningMechanism.output_values>` attribute.  It is
  also assigned as the `sender <LearningProjection.sender>` of the `LearningProjection` that projects to the
  `MappingProjection` being learned.

  **Multiple LearningSignals and LearningProjections.** Though not common, it is possible for a LearningMechanism to
  have more than one LearningSignal, and/or its LearningSignal(s) to have more than one LearningProjection. This allows
  the learning of multiple MappingProjections to be governed by a single LearningMechanism; note, however, that all will
  use the same `learning_signal <LearningMechanism.learning_signal>` (this can be useful, for example, in implementing
  certain forms of `convolutional neural networks <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_
  If all of the LearningProjections are used to implement the same form of `modulation <ModulatorySignal_Modualtion>`,
  (determined by their LearningSignals' `modulation <LearningSignal.modulation>` attribute), then a single
  LearningSignal should be assigned multiple LearningProjections;  if different forms of modulation are required,
  then multipole LearningSignals should be implemented (one for each type of modulation), and LearningProjection(s)
  assigned to the relevvant LearningSignals.  Multiple LearningSignals can be specified for a LearningMechanism
  by including them in a list assigned to the **learning_signals** argument of the LearningMechanism's
  constructor.  Each LearningSignal can be assigned multiple LearningProjections in the **projections** argument of
  its constructor, or the *PROJECTIONS* entry of a dictionary assigned to its **params** argument); however, the
  `matrix <MappingProjection.matrix>` parameter for all of them must have the same shape. The `learning_rate
  <LearningSignal.learning_rate>` for each LearningSignal, and the `learning_rate <LearningProjection.learning_rate>`
  for each LearningProjection of a LearningSignal, call all be assigned different values (with the latter taking
  precedence over the former).  If none of these are specified, the `learning_rate <LearningMechanism.learning_rate>`
  of the LearningMechanism is used. All of the LearningSignals of a LearningMechanism are listed in its
  `learning_signals` attribute (in addition to its `output_states <LearningMechanism.output_states>` attribute),
  and all of their LearningProjections are listed in the LearningMechanism's `learning_projections` attribute.

.. _LearningMechanism_Additional_Attributes:

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

In addition to its `input_states <LearningMechanism.input_states>`
States <State>` and `function <LearningMechanism.function>`, a LearningMechanism has the following
additonal attributes that refer to the Components being learned and/or its operation:

* `learned_projection` - the MappingProjection for which the LearningMechanism is responsible;  that is, the one with
  the `matrix <MappingProjection.matrix>` parameter that the LearningMechanism modifies;
..
* `error_source` - the Mechanism that receives the `learned_projection`;  that is, the one that generates the output
  used to calculate the error signal that the LearningMechanism attempts to reduce.
..
* `learning_projections` - the list of LearningProjections for all of the LearningMechanism's LearningSignals;
..
* `modulation` - this specifies the way in which the `learning_signal <LearningMechanism.learning_signal>` is used
  to modify the `matrix <MappingProjection.matrix>` parameter of the `learned_projection`.  By default its value is
  Modulation.ADD, which causes the weight changes in the `learning_signal` to be added to the current value of the
  `matrix <MappingProjection.matrix>` parameter (see `LearningMechanism_Execution` for a description of how the
  modifications are executed).
..
.. _LearningMechanism_Learning_Rate:

* `learning_rate <LearningMechanism.learning_rate>` - the learning rate for the LearningMechanism.  This is used to
  specify the :keyword:`learning_rate` parameter for its `function <LearningMechanism.function>`.  In general, the
  `learning_rate <LearningMechanism.learning_rate>` multiplies the weight changes provided by the LearningMechanism to
  its `function <LearningMechanism.function>` before conveying these to the `LearningSignal` used to modify the
  MappingProjection's `matrix <MappingProjection.matrix>` parameter. Specifying the
  `learning_rate <LearningMechanism.learning_rate>` for LearningMechanism (or the :keyword:`learning_rate` parameter
  of its `function <LearningMechanism.function>` directly) supersedes any specification of the :keyword:`learning_rate`
  for any `Process <Process.Process_Base.learning_rate>` and/or `System <System.System_Base.learning_rate>` to which
  the LearningMechanism belongs.  The default value for a LearningMechanism's `learning_rate <LearningMechanism>`
  attribute is `None`, in which case the LearningMechanism (and its `function <LearningMechanism.function>`) inherit
  the specification of the `learning_rate <Process.Process_Base.learning_rate>` for the Process in which the
  LearningMechanism is executed. If that is `None`, then it inherits it from the System in which it is executed.  If
  that is also `None`, then it uses the default value assigned by its `function <LearningMechanism.function>`.
  Learning rate can also be specified individually for `LearningSignals <LearningSignal>` and/or their associated
  `LearningProjections <LearningProjection>`.  Those have a direct multiplicative effect on the
  LearningProjection's `learning_signal <LearningProjection.learning_signal>` used to modify the weight matrix of the
  `learning_projection <LearningMechanism.learned_projection>`
  (see `LearningSignal learning_rate <LearningSignal_Learning_Rate>` for additional details).

COMMENT:
@@@ THE FOLLOWING SECTION SHOULD BE MOVED TO THE "USER'S MANUAL" WHEN THAT IS WRITTEN
COMMENT
.. _LearningMechanism_Learning_Configurations:

Learning Configurations
~~~~~~~~~~~~~~~~~~~~~~~

When learning is specified for a `MappingProjection`, a `Process <Process_Learning>`, or a
`System <System_Execution_Learning>`, PsyNeuLink automatically creates all of the components required for the
`MappingProjections <MappingProjection>` between `ProcessingMechanisms <ProcessingMechanism>` in that composition to
be learned.  The type of components that are generated depends on the :ref:`learning function <LearningFunction>`
specified, and the configuration of the composition.  All of the learning components of a System can be displayed
using the System's `show_graph` method with its **show_learning** argument assigned `True`.

.. _LearningMechanism_Single_Layer_Learning:

Single layer learning
^^^^^^^^^^^^^^^^^^^^^

This is the case when only a single MappingProjection is specified for learning, or the LearningMechanism's function
only considers the output of its `error_source <LearningMechanism_Additional_Attributes>`  when computing the changes
that will be made to the `learned_projection's  <LearningMechanism_Additional_Attributes>`
`matrix <MappingProjection.matrix>` (e.g., `Reinforcement`).  In this case, a single `ObjectiveMechanism` and
LearningMechanism are created for the `learned_projection <LearningMechanism_Additional_Attributes>`, if they do not
already exist, along with the following MappingProjections:

* from an OutputState of the LearningMechanism's `error_source <LearningMechanism_Additional_Attributes>` to the
  ObjectiveMechanism's `SAMPLE` :ref:`InputState <LINK>`.  By default, the
  `primary OutputState <OutputState_Primary>` of the error_souce is used;
  however, this can be modified by specifying its `MONITOR_FOR_LEARNING` parameter
  (see `above <LearningMechanism_Activation_Output>`).

* from the Process or System to the ObjectiveMechanism's `TARGET` :ref:`InputState <LINK>`;

* from the ObjectiveMechanism's `primary OutputState <OutputState_Primary>` to the LearningMechanism's
  `ERROR_SIGNAL <LearningMechanism_Activation_Input>` InputState .

In addition, a `LearningProjection` is created from the LearningMechanism's
`LEARNING_SIGNAL <LearningMechanism_Learning_Signal>` OutputState to the `matrix` `ParameterState`
for the `learned_projection <LearningMechanism_Additional_Attributes>`.  Because this case involves only a single
layer of learning, *no* Projection is created or assigned to the LearningMechanism's
`ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` OutputState.

.. _LearningMechanism_Single_Layer_Learning_Learning_Figure:

    **Components for Single Layer Learning**

    .. figure:: _static/LearningMechanism_Single_Layer_Learning_fig.svg
       :alt: Schematic of Mechanisms and Projections involved in learning for a single MappingProjection
       :scale: 50%

       ObjectiveMechanism, LearningMechanism and associated Projections created for a single learned_projection
       and error_source.  Each Mechanism is labeled by its type (upper line, in bold) and its designated
       status in the Process and/or System to which it belongs (lower line, caps).  Italicized labels beside a
       component indicates the attribute of the LearningMechanism with which it is associated.


.. _LearningMechanism_Multilayer_Learning:

Multilayer learning
^^^^^^^^^^^^^^^^^^^

This is the case when a set of MappingProjections are being learned that are in a sequence (such as the `pathway` of a
Process); that is, in which each projects to a ProcessingMechanism that is the `sender <MappingProjection.sender>` for
the next MappingProjection in the sequence (see the `figure <LearningMechanism_Multilayer_Learning_Figure>` below).
This requires the use of a learning function that can calculate the influence that each MappingProjection and its input
have on the error that the LearningMechanism receives from the next one in the sequence (e.g., `BackPropagation`).
In multilayer learning, the components created depend on the position of the
`learned_projection <LearningMechanism_Additional_Attributes>` and
`error_source <LearningMechanism_Additional_Attributes>` in the sequence.  If these are the last ones in the
sequence, they are treated in the same way as `single layer learning <LearningMechanism_Single_Layer>`.  This is the
case if the `error_source` is a standalone Mechanism (one not in a Process or System), the `TERMINAL` Mechanism of a
standalone Process (i.e., one not in a System), or the `TERMINAL` of all of the Processes to which it belongs in a
System (and therefore a `TERMINAL` for the System).  In these cases, as for single layer learning,
an `ObjectiveMechanism` is created that receives the output of the `error_source` as well as the target for learning
(see `LearningMechanisms_Targets` below), and projects to a LearningMechanism that is created for the
`learned_projection`.  For all others, the following MappingProjections are created (shown in the `figure
<LearningMechanism_Multilayer_Learning_Figure>` below):

* from the `sender <MappingProjection.sender>` of the `learned_projection` to the LearningMechanism's
  `ACTIVATION_INPUT` `InputState <LearningMechanism_Activation_Input>`.
..
* from the `error_source` to the LearningMechanism's
  `ACTIVATION_OUTPUT` `InputState <LearningMechanism_Activation_Output>`.
..
* from the `ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` OutputState of the LearningMechanism for the
  next MappingProjection in the sequence (i.e., the layer "above" it) to the LearningMechanism's
  `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` InputState.

In addition, a `LearningProjection` is created from each LearningMechanism's
`LEARNING_SIGNAL <LearningMechanism_Learning_Signal>` OutputState to the `matrix` `ParameterState`
of its `learned_projection`.  If the `learned_projection` is the first in the sequence, then *no* Projection is
created or assigned to its LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` OutputState.

.. _LearningMechanism_Multilayer_Learning_Figure:

    **Components for Multilayer Learning**

    .. figure:: _static/LearningMechanism_Multilayer_Learning_fig.svg
       :alt: Schematic of Mechanisms and Projections involved in learning for a sequence of MappingProjections
       :scale: 50%

       ObjectiveMechanism and LearningMechanisms and associated Projections created for a sequence of two
       MappingProjections specified for learning.  Each Mechanism is labeled by its type (uppler line, in bold) and
       its designated status in the Process and/or System to which it belongs (lower line, caps).  Italicized labels
       beside a component indicates the attribute of the LearningMechanism with which it is associated.

.. _LearningMechanism_Targets:

**TARGET Mechanisms**: receive the targets specified for learning.  When learning is specified for a `Process
<Process_Learning>` or `System <System_Execution_Learning>`, the `ObjectiveMechanism`  that will receive its
`targets <Run_Targets>` (specified in the call to its :keyword:`execute` or :keyword:`run` method) are identified and
designated as `TARGET` Mechanisms. These are listed in the Process' or System's `target_mechanisms` attribute.
It is important to note that the status of a `ProcessingMechanism` in a System takes precedence over its status in any
of the Processes to which it belongs. This means that even if a Mechanism is the `TERMINAL` of a particular Process,
if that Process is combined with others in a System, the Mechanism appears in any of those other Processes,
and it is not the `TERMINAL` of all of them, then it will *not* be the `TERMINAL` for the System.  As consequence,
although it will project to a `TARGET` Mechanism in the Process for which it is the `TERMINAL`, it will not do so in
the System (see :ref:`figure below <LearningProjection_Target_vs_Terminal_Figure>` for an example).  Finally, if a
Mechanisms is the `TERMINAL` for more than one Process used to create a System (that is, the pathways for those
Processes converge on that Mechanism), only one ObjectiveMechanism will be created for it in the System.

.. _LearningProjection_Target_vs_Terminal_Figure:

    **TERMINAL** and **TARGET** Mechanisms in Learning

    .. figure:: _static/LearningMechanism_TERMINAL_vs_TARGET_fig.svg
       :alt: Schematic of Mechanisms and Projections involved in learning
       :scale: 50 %

       Mechanism 3 is the `TERMINAL` Mechanism for Process A, However, it is also an `INTERNAL` Mechanism of Process B.
       Therefore, Mechanism 3 is designated as an `INTERNAL` Mechanism for the System, and Mechanism 4 is its `TERMINAL`
       Mechanism. As a consequence, if `BackPropagation` is used for learning, then Mechanism 4 is an
       `ObjectiveMechanism` and designated as a `TARGET`, while Mechanism 3 is a LearningMechanism
       and designated as a `LEARNING` Mechanism.

.. _LearningMechanism_Execution:

Execution
---------

LearningMechanisms are executed after all of the ProcessingMechanisms in the Process or System to which it belongs have
been executed, including the ObjectiveMechanism(s) that provide an error signal to the LearningMechanism(s).  When a
LearningMechanism is executed, it uses the value of its `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>`
InputState to calculate changes to the `matrix <MappingProjection.MappingProjection.matrix>` of the MappingProjections
being learned.  That value is assigned to its `learning_signal` attribute, as the value of each of the LearningSignals
in its `learning_signal` attribute, and as the value of each of their LearningProjections.  That value is used,
in turn, to modify the value of the `MATRIX` ParameterState of each of the MappingProjections being learned
(listed in the LearningMechanism's `learned_projections` attribute).  Each ParameterState uses the value it receives
from the `LearningProjection` to modify the parameter of its function in a manner specified by the
`modulation <LearningSignal.modulation>` attribute of the `LearningSignal` from which it receives the
LearningProjection (see `modulation <ModulatorySignals_Modulation>` for a description of State value modulation).
By default, the `modulation <LearningSignal.modulation>` attribute of a LearningSignal is Modulation.ADD,
the `function <ParameterState.function>` of a `MATRIX` ParameterState for a MappingProjection is
`Accumulator`, and the parameter it uses for `additive modulation` is its `increment <Accumulator.increment>`.  These
assignments cause the value of a LearningProjection to be added to the previous value of the `MATRIX` ParameterState,
thus incrementing the weights by the `learning_signal` specified by the LearningMechanism.  Note, however, that these
changes are not applied to the `matrix <MappingProjection.MappingProjection.matrix>` itself until the next
time the `learned_projection` is executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).
In addition to computing and conveying its `learning_signal`, the LearningMechanism's
`function <LearningMechanism.function>` also computes an error signal that is assigned to its
`error_signal <LearningMechanism.error_signal>` attribute and as the value of its *ERROR_SIGNAL* OutputState.

.. _LearningMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import parameter_keywords
from PsyNeuLink.Components.Functions.Function \
    import BackPropagation, ModulationParam, _is_modulation_param, is_function_type
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism \
    import ERROR_SIGNAL, ObjectiveMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection \
    import Projection_Base, _is_projection_spec, _validate_receiver, projection_keywords
from PsyNeuLink.Components.ShellClasses import Mechanism, Projection
from PsyNeuLink.Globals.Keywords import CONTROL_PROJECTIONS, DEFERRED_INITIALIZATION, FUNCTION_PARAMS, \
    IDENTITY_MATRIX, INDEX, INITIALIZING, INPUT_STATES, \
    LEARNED_PARAM, LEARNING, LEARNING_MECHANISM, LEARNING_PROJECTION, \
    LEARNING_SIGNAL, LEARNING_SIGNALS, LEARNING_SIGNAL_SPECS, \
    MAPPING_PROJECTION, MATRIX, NAME, OUTPUT_STATES, PARAMETER_STATE, PARAMS, PROJECTION, PROJECTIONS
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList, is_numeric, parameter_spec
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

# Params:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return `True` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
    Otherwise, return :keyword:`False`

    """
    if spec is LEARNING:
        return True
    else:
        return _is_projection_spec(spec)


# Used to index variable:
ACTIVATION_INPUT_INDEX = 0
ACTIVATION_OUTPUT_INDEX = 1
ERROR_OUTPUT_INDEX = 2
ERROR_SIGNAL_INDEX = 3

# Used to name input_states:
ACTIVATION_INPUT = 'activation_input'     # InputState
ACTIVATION_OUTPUT = 'activation_output'   # InputState

input_state_names =  [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]
output_state_names = [ERROR_SIGNAL, LEARNING_SIGNAL]

ERROR_SOURCE = 'error_source'

DefaultTrainingMechanism = ObjectiveMechanism

class LearningMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningMechanism(AdaptiveMechanism_Base):
    """
    LearningMechanism(                             \
        variable,                                  \
        error_source                               \
        function=BackPropagation                   \
        learning_rate=None                         \
        learning_signals=LEARNING_SIGNAL,          \
        modulation=ModulationParam.MULTIPLICATIVE  \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Implements a Mechanism that modifies the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.

    COMMENT:
        Description:
            LearningMechanism is a subtype of the AdaptiveMechanism Type of the Mechanism Category of Component
            It implements a Mechanism that calculates changes to a Projection's parameters.
            It's function takes the output of an ObjectiveMechanism (self.variable) and generates a
            learning_signal (ndarray of parameter changes) to be used by the recipient of a LearningProjection
            that projects from the LearningMechanism to a MappingProjection.

        # DOCUMENT: ??NOT SURE WHETHER THIS IS STILL RELEVANT
        #    IF objective_mechanism IS None, IT IS LEFT UNSPECIFIED (FOR FURTHER IMPLEMENTATION BY COMPOSITION)
        #    THESE ARE HANDLED BY A MODULE METHOD _instantiate_objective_mechanism (AS PER OBJECTIVE MECHANISM):
        #        IF objective_mechanism IS SPECIFIED AS ObjectiveMechanism, AN OBJECTIVE MECHANISM IS CREATED FOR IT
        #        IF objective_mechanism IS SPECIFIED AS A MECHANISM OR OUTPUTSTATE,
        #               a MappingProjection WITH AN IDENTITY MATRIX IS IMPLEMENTED FROM IT TO THE LearningMechanism

        Learning function:
            Generalized delta rule:
            dE/dW  =          learning_rate   *    dE/dA          *       dA/dW             *    I
            weight = weight + (learning_rate  * error_derivative  *  activation_derivative  *  input)
            for sumSquared error fct =        (target - output)
            for logistic activation fct =                           output * (1-output)
            where:
                output = activity of output (target) units (higher layer)
                input = activity of sending units (lower layer)
            Needs:
            - activation_derivative:  get from FUNCTION of sample_activation_mechanism/receiver_mech
                                      assumes derivative of Logistic unless otherwise specified
            - error_derivative:  get from FUNCTION of error_source/next_level_mech;  but handled in ObjectiveMechanism

        Class attributes:
            + className = LEARNING_MECHANISM
            + componentType = ADAPTIVE_MECHANISM
            + paramClassDefaults (dict):
                + FUNCTION (Function): (default: BP)
                + FUNCTION_PARAMS:
                    + LEARNING_RATE (value): (default: 1)
            + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            None

        MechanismRegistry:
            All instances of LearningMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    variable : List or 2d np.array
        specifies a template for the three items required by the `function <LearningMechanism.function>`: the input
        to the `learned_projection`, the output of the `error_source`, and the error_signal received by the
        LearningMechanism (see `variable <LearningMechanism.variable>` for details).

    error_source : ProcessingMechanism
        specifies the Mechanism the output of which is used to generate the error_signal received by the
        LearningMechanism (in its `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` InputState).

    learning_signals : List[parameter of Projection, ParameterState, Projection, tuple[str, Projection] or dict]
        specifies the parameter(s) to be trained by the LearningMechanism
        (see `learning_signals <LearningMechanism.learning_signals>` for details).

    modulation : ModulationParam : ModulationParam.ADDITIVE
        specifies the default form of modulation used by the LearningMechanism's LearningSignals,
        unless they are `individually specified <LearningSignal_Specification>`.

    function : LearningFunction or function
        specifies the function used to compute the `learning_signal` used by a LearningProjection, and the
        and `error_signal` passed to the next LearningMechanism in a
        `learning sequence <LearningMechanism_Learning_Configurations>`
        (see `function <LearningMechanism.function>` for details).

    learning_rate : float
        specifies the learning rate for this LearningMechanism
        (see `learning_rate <LearningMechanism.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default LearningProjection-<index>
        a string used for the name of the LearningProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the `PreferenceSet` for the LearningProjection.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    componentType : LEARNING_PROJECTION

    variable : 2d np.array
        specifies three items: 1) the input to the `learned_projection`; 2) the output of the `error_source`; and
        the error signal received from either an ObjectiveMechanism or the next LearningMechanism in a
        `learning sequence <LearningMechanism_Learning_Configurations>`.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    COMMENT:
      error_output : 1d np.array
            the output of the next Mechanism in the pathway (the one to which the `error_signal` pertains, and projected
            to by the Mechanism that receives the Projection being learned). Typically this comes from  the
            `LearningMechanism` for that next Mechanism.  However, if the current LearningMechanism is for the last
            Mechanism in a sequence of Mechanisms being learned (often, but not necessarily a `TERMINAL` Mechanism),
            then error_output is set to an array of 1's with a length equal to the length of the `error_signal`.

        error_source : ObjectiveMechanism or LearningMechanism
            the Mechanism from which the LearningMechanism gets its `error_signal`.  The LearningMechanism receives a
            Projection from the `error_source` to its `ERROR_SIGNAL <LearningMechanism.input_states>` InputState.
            If the `error_source` is an ObjectiveMechanism, the Projection is from its
            `primary OutputState <OutputState_Primary>`.  If the `error_source` is another LearningMechanism,
            the Projection is from its `ERROR_SIGNAL <LearningMechanism.outputStates>` OutputState.  In either case,
            the MappingProjection uses an `IDENTITY_MATRIX`, and so the value of the OutputState used for the
            `error_source` must be equal in length to the value of the LearningMechanism's `ERROR_SIGNAL` inputstate.
    COMMENT

    input_states : ContentAddressableList[OutputState]
        list containing the LearningMechanism's three `InputStates <LearningMechanism_InputStates>`:
        *ACTIVATION_INPUT*,  *ACTIVATION_OUTPUT*, and *ERROR_SIGNAL*.

    error_source : ProcessingMechanism
        the `Mechanism` responsible for the output used to generate the error signal received by the LearningMechanism
        (in its `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` InputState).

    learned_projections : List[Projections]
        the Projections, the parameters of which are modified by the LearningMechanism.  Usually this is a single
        `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.

    function : LearningFunction or function : default BackPropagation
        specifies function used to compute the `learning_signal`.  Must take the following arguments:
        `input` (list or 1d array), `output` (list or 1d array), `derivative` (function) and `error` (list or 1d array).
        See `LearningMechanism_Function` for additional details.

    learning_rate : float : None
        determines the learning rate for the LearningMechanism.  It is used to specify the :keyword:`learning_rate`
        parameter for the LearningMechanism's `learning function <LearningMechanism.function>`
        (see description of `learning_rate <LearningMechanism_Learning_Rate>` above for additional details).

    error_signal : 1d np.array
        one of two values returned by the LearningMechanism's `function <LearningMechanism.function>`.  For
        `single layer learning <LearningMechanism_Single_Layer>`, this is the same as the value received in the
        LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` InputState;  for `multilayer
        learning <LearningMechanism_Multilayer_Learning>`, it is a modified version of the value received, that takes
        account of the contribution to the error signal received, made by the learned_projection and its input.

    learning_signal : number, ndarray or matrix
        one of two values returned by the LearningMechanism's `function <LearningMechanism.function>`,
        that provides the changes to the `matrix <MappingProjection.matrix>` parameter(s) of the
        `MappingProjection(s) <MappingProjection>` being learned (and listed in
        `learned_projections <LearningMechanism.learned_projections>`) required to reduce the
        `error_signal <LearningMechanism.error_signal>`.

    learning_signals : List[LearningSignal]
        list of `LearningSignals <LearningSignals>` for the LearningMechanism, each of which sends a
        `LearningProjection` to the `ParameterState` for the parameter it controls.
        The value of each LearningSignal generally contains a 2d np.array or matrix of changes to be used by recipient
        of a `LearningProjection` from the LearningMechanism, to adjust its parameters (e.g., matrix weights, in which
        rows correspond to sender, and columns to receiver).  Since LearningSignals are OutputStates, these are also
        listed in the ControlMechanism's `output_states <Mechanism.output_states>` attribute, along with its
        *ERROR_SIGNAL* OutputState, and any others it may have.

    output_states : ContentAddressableList[OutputState]
        contains list of OutputStates for the LearningMechanism, including: its LearningSignal(s) which appear(s) at
        the begining of the list; its *ERROR_SIGNAL* OutputState, which appears after the LearningSignal(s); any
        additional (e.g., user-specified) OutputStates, which appear at the end of the list.

    #  FIX: THIS MAY NEED TO BE A 3d array (TO ACCOMDOATE 2d array (MATRICES) AS ENTRIES)
    output_values : 2d np.array
        the initial item(s) is/are the value(s) of the LearningMechanism's LearningSignal(s);  the next is
        the value of its *ERROR_SIGNAL* OutputState (same as `error_signal`);  subsequent items are the value of
        the corresponding OutputStates in the `output_states <Mechanism_Base.outputStates>` attribute.

    modulation : ModulationParam
        the default form of modulation used by the LearningMechanism's `LearningSignals <LearningSignal>`,
        unless they are `individually specified <LearningSignal_Specification>`.

    name : str : default LearningProjection-<index>
        the name of the LearningMechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for Projection.
        Specified in the **prefs** argument of the constructor for the Projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = LEARNING_MECHANISM
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # variableClassDefault = None

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        CONTROL_PROJECTIONS: None,
        INPUT_STATES:input_state_names,
        OUTPUT_STATES:[{NAME:ERROR_SIGNAL,
                        INDEX:1},
                       {NAME:LEARNING_SIGNAL,  # NOTE: This is the default, but is overridden by any LearningSignal arg
                        INDEX:0}
                       ]})

    @tc.typecheck
    def __init__(self,
                 variable:tc.any(list, np.ndarray),
                 size=None,
                 error_source:tc.optional(Mechanism)=None,
                 function:is_function_type=BackPropagation,
                 learning_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.ADDITIVE,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  learning_signals=learning_signals,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name
        # delete self.init_args[ERROR_SOURCE]

        # # Flag for deferred initialization
        # self.value = DEFERRED_INITIALIZATION

        self._learning_rate = learning_rate

        super().__init__(variable=variable,
                         size=size,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):
        """Validate that variable has exactly three items: activation_input, activation_output and error_signal
        """

        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise LearningMechanismError("Variable for {} ({}) must have three items ({}, {}, and {})".
                                format(self.name, self.variable,
                                       ACTIVATION_INPUT,
                                       ACTIVATION_OUTPUT,
                                       ERROR_SIGNAL))

        # Validate that activation_input, activation_output, and error_signal are numeric and lists or 1d np.ndarrays
        for i in range(len(self.variable)):
            item_num_string = ['first', 'second', 'third'][i]
            item_name = input_state_names[i]
            if not np.array(self.variable[i]).ndim == 1:
                raise LearningMechanismError("The {} item of variable for {} ({}:{}) is not a list or 1d np.array".
                                              format(item_num_string, self.name, item_name, self.variable[i]))
            if not (is_numeric(self.variable[i])):
                raise LearningMechanismError("The {} item of variable for {} ({}:{}) is not numeric".
                                              format(item_num_string, self.name, item_name, self.variable[i]))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate error_source as an Objective Mechanism or another LearningMechanism
        """

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

        if ERROR_SOURCE in target_set:
            if not isinstance(target_set[ERROR_SOURCE], (ObjectiveMechanism, LearningMechanism)):
                raise LearningMechanismError("{} arg for {} must be an ObjectiveMechanism or another LearningMechanism".
                                             format(ERROR_SOURCE, self.name))


        # FIX: REPLACE WITH CALL TO _parse_state_spec WITH APPROPRIATE PARAMETERS
        if LEARNING_SIGNALS in target_set and target_set[LEARNING_SIGNALS]:

            from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal \
                import LearningSignal
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

            for spec in target_set[LEARNING_SIGNALS]:

                learning_proj = None  # Projection from LearningSignal to MappingProjection
                mapping_proj = None   # MappingProjection that receives Projection from LearningSignal

                # Specification is for a LearningSignal
                if isinstance(spec, LearningSignal):
                    #  Check that any LearningProjections it has
                    #    are to MappingProjections to Mechanisms in the same System
                    for learning_proj in spec.efferents:
                        _validate_receiver(self,learning_proj, MappingProjection, LEARNING_SIGNAL, context)
                    continue

                # Specification is for a ParameterState
                elif isinstance(spec, ParameterState):
                    param_name = spec.name
                    mapping_proj = spec.owner

                # Specification is for a Projection
                elif isinstance(spec, Projection):
                    if isinstance(spec, LearningProjection):
                        param_name = spec.receiver.name
                        learning_proj = spec
                        mapping_proj = learning_proj.receiver.owner
                    elif isinstance(spec, MappingProjection):
                        param_name = MATRIX
                        mapping_proj = spec
                    else:
                        raise LearningMechanismError("The {} specified in the {} arg for {} ({}) must be a {}".
                                                     format(PROJECTION,
                                                            LEARNING_SIGNALS,
                                                            self.name,
                                                            spec.name,
                                                            MAPPING_PROJECTION))

                # Specification is for a tuple (str, MappingProjection):
                elif isinstance(spec, tuple):
                    param_name = spec[0]
                    mapping_proj = spec[1]
                    # Check that 1st item is a str (presumably the name of the learned Projection's attribute
                    #    for the param to be learned; e.g., 'MATRIX' for MappingProjection)
                    if not isinstance(param_name, str):
                        raise LearningMechanismError("1st item of tuple in specification of {} for {} ({}) "
                                                     "must be a string".format(LEARNING_SIGNAL, self.name, param_name))
                    # Check that 2nd item is a MappingProjection
                    if not isinstance(mapping_proj, MappingProjection):
                        raise LearningMechanismError("2nd item of tuple in specification of {} for {} ({}) "
                                                     "must be a {}".
                                                     format(LEARNING_SIGNAL,
                                                            self.name,
                                                            mapping_proj,
                                                            MAPPING_PROJECTION))

                # LearningSignal specification dictionary, must have the following entries:
                #    NAME:str - must be the name of an attribute of PROJECTION
                #    PROJECTION:Projection - must be a MappingProjection
                #                            and have an attribute and corresponding ParameterState named NAME
                #    PARAMS:dict - entries must be valid LearningSignal parameters (e.g., LEARNING_RATE)
                elif isinstance(spec, dict):
                    if not NAME in spec:
                        raise LearningMechanismError("Specification dict for {} of {} must have a {} entry".
                                                    format(LEARNING_SIGNAL, self.name, NAME))
                    param_name = spec[NAME]
                    if not PROJECTION in spec:
                        raise LearningMechanismError("Specification dict for {} of {} must have a {} entry".
                                                    format(LEARNING_SIGNAL, self.name, PROJECTION))
                    mapping_proj = spec[PROJECTION]
                    if not isinstance(mapping_proj, MappingProjection):
                        raise LearningMechanismError("{} entry of specification dict for {} of {} must be a {}".
                                                    format(PROJECTION, LEARNING_SIGNAL, self.name, MAPPING_PROJECTION))
                    # Check that all of the other entries in the specification dictionary
                    #    are valid LearningSignal params
                    for param in spec:
                        if param in {NAME, PROJECTION}:
                            continue
                        if not hasattr(LearningSignal, param):
                            raise LearningMechanismError("Entry in specification dictionary for {} arg of {} ({}) "
                                                       "is not a valid {} parameter".
                                                       format(LEARNING_SIGNAL, self.name, param,
                                                              LearningSignal.__class__.__name__))
                else:
                    raise LearningMechanismError("PROGRAM ERROR: unrecognized specification for {} arg of {} ({})".
                                                format(LEARNING_SIGNALS, self.name, spec))
                    # raise LearningMechanismError("Specification of {} for {} ({}) must be a "
                    #                             "ParameterState, Projection, a tuple specifying a parameter and "
                    #                              "Projection, a LearningSignal specification dictionary, "
                    #                              "or an existing LearningSignal".
                    #                             format(CONTROL_SIGNAL, self.name, spec))

                # Validate that the receiver of the LearningProjection (if specified)
                #     is a MappingProjection and in the same System as self (if specified)
                if learning_proj:
                    _validate_receiver(sender_mech=self,
                                       projection=learning_proj,
                                       expected_owner_type=MappingProjection,
                                       spec_type=LEARNING_SIGNAL,
                                       context=context)

                # IMPLEMENTATION NOTE: the tests below allow for the possibility that the MappingProjection
                #                      may not yet be fully implemented (e.g., this can occur if the
                #                      LearningMechanism being implemented here is as part of a LearningProjection
                #                      specification for the MappingProjection's matrix param)
                # Check that param_name is the name of a parameter of the MappingProjection to be learned
                if not param_name in (set(mapping_proj.user_params) | set(mapping_proj.user_params[FUNCTION_PARAMS])):
                    raise LearningMechanismError("{} (in specification of {} for {}) is not an "
                                                "attribute of {} or its function"
                                                .format(param_name, LEARNING_SIGNAL, self.name, mapping_proj))
                # Check that the MappingProjection to be learned has a ParameterState for the param
                if mapping_proj._parameter_states and not param_name in mapping_proj._parameter_states.names:
                    raise LearningMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                "specified in {} for {}".
                                                format(param_name, mapping_proj.name, LEARNING_SIGNAL, self.name))

    def _instantiate_attributes_before_function(self, context=None):
        """Instantiates MappingProjection from error_source (if specified) to the LearningMechanism

        Also assigns learned_projection attribute (to MappingProjection being learned)
        """

        super()._instantiate_attributes_before_function(context=context)

        if self.error_source:
            _instantiate_error_signal_projection(sender=self.error_source, receiver=self)

    def _instantiate_output_states(self, context=None):

        # Create registry for LearningSignals (to manage names)
        from PsyNeuLink.Globals.Registry import register_category
        from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
        from PsyNeuLink.Components.States.State import State_Base
        register_category(entry=LearningSignal,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        # Instantiate LearningSignals if they are specified, and assign to self._output_states
        # Note: if any LearningSignals are specified they will replace the default LEARNING_SIGNAL OutputState
        #          in the OUTPUT_STATES entry of paramClassDefaults;
        #       the LearningSignals are appended to _output_states, leaving ERROR_SIGNAL as the first entry.
        if self.learning_signals:
            # Delete default LEARNING_SIGNAL item in output_states
            del self._output_states[1]
            for i, learning_signal in enumerate(self.learning_signals):
                # Instantiate LearningSignal
                ls = self._instantiate_learning_signal(learning_signal=learning_signal, context=context)
                # Add LearningSignal to ouput_states list
                self._output_states.append(ls)
                # Replace spec in learning_signals list with actual LearningSignal
                self.learning_signals[i] = ls

        super()._instantiate_output_states(context=context)

    def _instantiate_learning_signal(self, learning_signal=None, context=None):
        """Instantiate LearningSignal OutputState and assign (if specified) or instantiate LearningProjection

        Notes:
        * learning_signal arg can be a:
            - LearningSignal object;
            - LearningProjection;
            - ParameterState;
            - Projection (in which case, MATRIX parameter is used as receiver of LearningProjection)
            - params dict containing a LearningProjection;
            - tuple (param_name, PROJECTION), from learning_signals arg of constructor;
                    [NOTE: this is a convenience format;
                           it precludes specification of LearningSignal params (??e.g., LEARNING_RATE)]
            - LearningSignal specification dictionary, from learning_signals arg of constructor
                    [NOTE: this must have at least NAME:str (param name) and MECHANISM:Mechanism entries;
                           it can also include a PARAMS entry with a params dict containing LearningSignal params]
            * NOTE: ParameterState must be for a Projection, and generally for MATRIX parameter of a MappingProjection;
                    however, LearningSignal is implemented to be applicable for any ParameterState of any Projection.
        * State._parse_state_spec() is used to parse learning_signal arg
        * params are expected to be for (i.e., to be passed to) LearningSignal;
        * wait to instantiate deferred_init() Projections until after LearningSignal is instantiated,
             so that correct OutputState can be assigned as its sender;
        * index of OutputState is incremented based on number of LearningSignals already instantiated;
            this means that a LearningMechanism's function must return as many items as it has LearningSignals,
            with each item of the function's value used by a corresponding LearningSignal.
            NOTE: multiple LearningProjections can be assigned to the same LearningSignal to implement "ganged" learning
                  (that is, learning of many Projections with a single value)

        Returns LearningSignal (OutputState)
        """

# FIX: THESE NEEDS TO BE DEALT WITH
# FIX: learning_projection -> learning_projections
# FIX: trained_projection -> learned_projection
# FIX: error_signal ??OR?? error_signals??
# FIX: LearningMechanism: learned_projection attribute -> learned_projections list
#                         learning_signal -> learning_signals (WITH SINGULAR ONE INDEXING INTO learning_signals.values)
#  FIX: THIS MAY NEED TO BE A 3d array (TO ACCOMDOATE 2d array (MATRICES) AS ENTRIES)

        from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
        from PsyNeuLink.Components.States.State import _parse_state_spec
        from PsyNeuLink.Components.States.ParameterState import ParameterState, _get_parameter_state
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

        # FIX: NEED TO CHARACTERIZE error_signal FOR BELOW
        # # EXTEND error_signals TO ACCOMMODATE NEW LearningSignal -------------------------------------------------
        # #        also used to determine constraint on LearningSignal output value
        #
        # if not hasattr(self, ERROR_SIGNALS) or self.error_signals is None:
        #     self.error_signals = np.array(defaultErrorSignal)
        # else:
        #     self.error_signals = np.append(self.error_signals, defaultErrorSignal)

        # GET index FOR LearningSignal OutputState
        try:
            output_state_index = len(self.output_states)
        except (AttributeError, TypeError):
            output_state_index = 0


        # PARSE learning_signal SPECIFICATION -----------------------------------------------------------------------

        learning_projection = None
        mapping_projection = None
        learning_signal_params = None

        learning_signal_spec = _parse_state_spec(owner=self, state_type=LearningSignal, state_spec=learning_signal)

        # Specification is a ParameterState
        if isinstance(learning_signal_spec, ParameterState):
            mapping_projection = learning_signal_spec.owner
            if not isinstance(mapping_projection, MappingProjection):
                raise LearningMechanismError("{} specified for {} of {} ({}) must be a {}".
                                             format(PARAMETER_STATE,
                                                    LEARNING_SIGNAL,
                                                    self.name,
                                                    mapping_projection,
                                                    PROJECTION))
            param_name = learning_signal_spec.name
            parameter_state = _get_parameter_state(self, LEARNING_SIGNAL, param_name, mapping_projection)

        # Specification was tuple or dict, and parsed into a dict
        elif isinstance(learning_signal_spec, dict):
            param_name = learning_signal_spec[NAME]
            learning_signal_params = learning_signal_spec[PARAMS]

            # learning_signal was a specification dict, with PROJECTION as an entry (and parameter as NAME)
            if learning_signal_params and PROJECTION in learning_signal_params:
                mapping_projection = learning_signal_params[PROJECTION]
                # Delete PROJECTION entry as it is not a parameter of LearningSignal
                #     (which will balk at it in LearningSignal._validate_params)
                del learning_signal_params[PROJECTION]
                parameter_state = _get_parameter_state(self, LEARNING_SIGNAL, param_name, mapping_projection)

            # Specification was originally a tuple, either in parameter specification or learning_signal arg;
            #    1st item was either assigned to the NAME entry of the learning_signal_spec dict
            #        (if tuple was a (param_name, Projection tuple) for learning_signal arg;
            #        or used as param value, if it was a parameter specification tuple
            #    2nd item was placed in learning_signal_params entry of params dict in learning_signal_spec dict,
            #        so parse:
            # FIX 5/23/17: NEED TO GET THE KEYWORDS STRAIGHT FOR PASSING LearningSignal SPECIFICATIONS
            # IMPLEMENTATION NOTE:
            #    PROJECTIONS is used by _parse_state_spec to place the 2nd item of any tuple in params dict;
            #                      here, the tuple comes from a (param, Projection) specification in learning_signal arg
            #    Delete whichever one it was, as neither is a recognized LearningSignal param
            #        (which will balk at it in LearningSignal._validate_params)
            elif (learning_signal_params and
                    any(kw in learning_signal_spec[PARAMS] for kw in {LEARNING_SIGNAL_SPECS, PROJECTIONS})):
                if LEARNING_SIGNAL_SPECS in learning_signal_spec[PARAMS]:
                    spec = learning_signal_params[LEARNING_SIGNAL_SPECS]
                    del learning_signal_params[LEARNING_SIGNAL_SPECS]
                elif PROJECTIONS in learning_signal_spec[PARAMS]:
                    spec = learning_signal_params[PROJECTIONS]
                    del learning_signal_params[PROJECTIONS]

                # LearningSignal
                if isinstance(spec, LearningSignal):
                    learning_signal_spec = spec

                else:
                    # Projection
                    # IMPLEMENTATION NOTE: Projection was placed in list in PROJECTIONS entry by _parse_state_spec
                    if isinstance(spec, list) and isinstance(spec[0], Projection):
                        if isinstance(spec[0], MappingProjection):
                            mapping_projection = spec[0]
                            param_name = MATRIX
                            parameter_state = _get_parameter_state(self,
                                                                   LEARNING_SIGNAL,
                                                                   param_name,
                                                                   mapping_projection)
                        elif isinstance(spec[0], LearningProjection):
                            learning_projection = spec[0]
                            if learning_projection.value is DEFERRED_INITIALIZATION:
                                parameter_state = learning_projection.init_args['receiver']
                            else:
                                parameter_state = learning_projection.receiver
                            param_name = parameter_state.name
                        else:
                            raise LearningMechanismError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                        "must contain a single MappingProjection or LearningProjection".
                                                        format(LEARNING_SIGNAL_SPECS, learning_signal, self.name))

                        if len(spec)>1:
                            raise LearningMechanismError("PROGRAM ERROR: Multiple LearningProjections is not "
                                                        "currently supported in specification of a LearningSignal")
                    else:
                        raise LearningMechanismError("PROGRAM ERROR: failure to parse specification of {} for {}".
                                                    format(learning_signal, self.name))
            else:
                raise LearningMechanismError("PROGRAM ERROR: No entry found in params dict with specification of "
                                            "parameter Projection or LearningProjection for {} of {}".
                                            format(learning_signal, self.name))


        # INSTANTIATE LearningSignal -----------------------------------------------------------------------------------

        # Specification is a LearningSignal (either passed in directly, or parsed from tuple above)
        if isinstance(learning_signal_spec, LearningSignal):
            # Deferred Initialization, so assign owner, name, and initialize
            if learning_signal_spec.value is DEFERRED_INITIALIZATION:
                # FIX 5/23/17:  IMPLEMENT DEFERRED_INITIALIZATION FOR LearningSignal
                # CALL DEFERRED INIT WITH SELF AS OWNER ??AND NAME FROM learning_signal_dict?? (OR WAS IT SPECIFIED)
                # OR ASSIGN NAME IF IT IS DEFAULT, USING learning_signal_DICT??
                pass
            elif not learning_signal_spec.owner is self:
                raise LearningMechanismError("Attempt to assign LearningSignal to {} ({}) that is already owned by {}".
                                            format(self.name,
                                                   learning_signal_spec.name,
                                                   learning_signal_spec.owner.name))
            learning_signal = learning_signal_spec
            learning_signal_name = learning_signal_spec.name
            learning_projections = learning_signal_spec.efferents

            # IMPLEMENTATION NOTE:
            #    THIS IS TO HANDLE FUTURE POSSIBILITY OF MULTIPLE ControlProjections FROM A SINGLE LearningSignal;
            #    FOR NOW, HOWEVER, ONLY A SINGLE ONE IS SUPPORTED
            # parameter_states = [proj.recvr for proj in learning_projections]
            if len(learning_projections) > 1:
                raise LearningMechanismError("PROGRAM ERROR: list of ControlProjections is not currently supported "
                                            "as specification in a LearningSignal")
            else:
                learning_projection = learning_projections[0]
                parameter_state = learning_projection.receiver

        # Specification is not a LearningSignal, so create OutputState for it
        else:
            learning_signal_name = param_name + '_' + LearningSignal.__name__

            from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal \
                import LearningSignal
            from PsyNeuLink.Components.States.State import _instantiate_state

            # Get constraint for OutputState's value
            # - assume that LearningMechanism.value has only two items (learning_signal and error_signal)
            # - use learning_signal (stored in self.learning_signal) as value for all LearningSignals
            self._update_value(context=context)
            constraint_value = self.learning_signal
            learning_signal_params.update({LEARNED_PARAM:param_name})

            learning_signal = _instantiate_state(owner=self,
                                                state_type=LearningSignal,
                                                state_name=learning_signal_name,
                                                state_spec=constraint_value,
                                                state_params=learning_signal_params,
                                                constraint_value=constraint_value,
                                                constraint_value_name='Default control allocation',
                                                context=context)

        # VALIDATE OR INSTANTIATE LearningProjection(s) FROM LearningSignal  -------------------------------------------

        # Validate learning_projection (if specified) and get receiver's name
        if learning_projection:
            _validate_receiver(self, learning_projection, MappingProjection, LEARNING_SIGNAL,context=context)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
            if not isinstance(learning_projection, LearningProjection):
                raise LearningMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a LearningProjection, to LearningSignal of {}".
                                                  format(learning_projection, self.name))
            if learning_projection.value is DEFERRED_INITIALIZATION:
                learning_projection.init_args['sender']=learning_signal
                if learning_projection.init_args['name'] is None:
                    # FIX 5/23/17: CLEAN UP NAME STUFF BELOW:
                    learning_projection.init_args['name'] = LEARNING_PROJECTION + \
                                                   ' for ' + parameter_state.owner.name + ' ' + parameter_state.name
                learning_projection._deferred_init()
            else:
                learning_projection.sender = learning_signal

            # Add LearningProjection to list of LearningSignal's outgoing Projections
            # (note: if it was deferred, it just added itself, skip)
            if not learning_projection in learning_signal.efferents:
                learning_signal.efferents.append(learning_projection)

            # Add LearningProjection to LearningMechanism's list of LearningProjections
            try:
                self.learning_projections.append(learning_projection)
            except AttributeError:
                self.learning_projections = [learning_projection]

        return learning_signal

    def _instantiate_attributes_after_function(self, context=None):

        if self._learning_rate is not None:
            self.learning_rate = self._learning_rate

        super()._instantiate_attributes_after_function(context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute LearningMechanism function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # COMPUTE LEARNING SIGNAL (dE/dW):
        self.learning_signal, self.error_signal = self.function(variable=variable,
                                                                params=runtime_params,
                                                                context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        self.value = [self.learning_signal, self.error_signal]
        return self.value

    @property
    def learning_rate(self):
        return self.function_object.learning_rate

    @learning_rate.setter
    def learning_rate(self, assignment):
        self.function_object.learning_rate = assignment

    @property
    def learned_projections(self):
        return [lp.receiver.owner for lp in self.learning_projections]


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _instantiate_error_signal_projection(sender, receiver):
    """Instantiate a MappingProjection to carry an error_signal to a LearningMechanism

    Can take as the sender an `ObjectiveMechanism` or a `LearningMechanism`.
    If the sender is an ObjectiveMechanism, uses its `primary OutputState <OutputState_Primary>`.
    If the sender is a LearningMechanism, uses its `ERROR_SIGNAL <LearningMechanism.outputStates>` OutputState.
    The receiver must be a LearningMechanism; its `ERROR_SIGNAL <LearningMechanism.input_states>` InputState is used.
    Uses and IDENTITY_MATRIX for the MappingProjection, so requires that the sender be the same length as the receiver.

    """

    if isinstance(sender, ObjectiveMechanism):
        sender = sender.output_states[ERROR_SIGNAL]
    elif isinstance(sender, LearningMechanism):
        sender = sender.output_states[ERROR_SIGNAL]
    else:
        raise LearningMechanismError("Sender of the error signal Projection {} must be either "
                                     "an ObjectiveMechanism or a LearningMechanism".
                                     format(sender))

    if isinstance(receiver, LearningMechanism):
        receiver = receiver.input_states[ERROR_SIGNAL]
    else:
        raise LearningMechanismError("Receiver of the error signal Projection {} must be a LearningMechanism".
                                     format(receiver))

    if len(sender.value) != len(receiver.value):
        raise LearningMechanismError("The length of the OutputState ({}) for the sender ({}) of "
                                     "the error signal Projection does not match "
                                     "the length of the InputState ({}) for the receiver ({})".
                                     format(len(sender.value), sender.owner.name,
                                            len(receiver.value),receiver.owner.name))

    return MappingProjection(sender=sender,
                             receiver=receiver,
                             matrix=IDENTITY_MATRIX,
                             name = sender.owner.name + ' ' + ERROR_SIGNAL)
