# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningMechanism *******************************************************

# DOCUMENT:
#    IF objective_mechanism IS  None, IT IS LEFT UNSPECIFIED (FOR FURTHER IMPLEMENTATION BY COMPOSITION)
#    THESE ARE HANDLED BY A MODULE METHOD _instantiate_objective_mechanism (AS PER OBJECTIVE MECHANISM):
#        IF objective_mechanism IS SPECIFIED AS ObjectiveMechanism, AN OBJECTIVE MECHANISM IS CREATED FOR IT
#        IF objective_mechanism IS SPECIFIED AS A MECHANISM OR OUTPUTSTATE,
#               a MappingProjection WITH AN IDENTITY MATRIX IS IMPLEMENTED FROM IT TO THE LearningMechanism

"""
.. _LearningMechanism_Overview:

Overview
--------

A LearningMechanism is an `AdaptiveMechanism <AdaptiveMechanism>` that modifies a parameter (usually the `matrix
<MappingProjection.matrix>`) of a `MappingProjection`.  It's function takes an `error_signal` (usually the output of
an `ObjectiveMechanism` or another `LearningMechanism`) and generates a `learning_signal` that is used to modify
the MappingProjection by way of a `LearningProjection`.  A LearningMechanism can modify only MappingProjections that
link ProcessingMechanisms in the same `system <System>` or `process <Process>` to which it belongs.  The learning  
components of a system can be displayed using the system's `show_graph` method with its **show_learning** argument 
assigned :keyword:``True`.  LearningMechanisms are execute after all of the ProcessingMechanisms in that system or 
process have been executed.  If they belong to a system, they are executed before the  
`control components <ControlMechanism>` for that system have been executed.

COMMENT:
  @@@ SEARCH FOR LearningProjection_Automatic_Creation AND REPLACE WITH REFERENCE TO THIS LABEL:
COMMENT

COMMENT:
  AT PRESENT, LearningMechanisms SUPPORTS MODIFICATION OF ONLY A SINGLE MappingProjection;  FUTURE VERSIONS MAY
  ALLOW MODIFICATION OF MULTIPLE MappingProjections (USING MULTIPLE CORRESPONDING error_signals).
COMMENT

.. _LearningMechanism_Creation:

Creating a LearningMechanism
----------------------------

LearningMechanisms can be created in any of the ways that can be used to `create mechanisms <Mechanism_Creation>`.
More commonly, however, they are created automatically when:

* the learning attribute is specified for a :ref:`system <LINK>` or :ref:`process <LINK>`;
..
* a `LearningProjection` (or the keyword `LEARNING`) is specified in a
  `tuple specification <MappingProjection_Tuple_Specification>`  for a MappingProjection in the `pathway` of a process;
..
* a `LearningProjection` is created without specifying its `sender <LearningProjection.sender>` attribute.

In these instances, an `ObjectiveMechanism`, `LearningProjection <LearningProjection>`, and any additional projections
required to implement learning that do not already exist are also instantiated.  This is described below, under
`Learning Configurations <LearningMechanism_Learning_Configurations>`.

.. _LearningMechanism_Structure:

Structure
---------

A LearningMechanism has three `inputStates <InputState>`, a learning `function <LearningMechanism.function>`,
and two `outputStates <OutputState>` that are used to receive, compute, and transmit the information needed to modify
the MappingProjection for which it is responsible.

.. _LearningMechanism_InputStates:

InputStates
~~~~~~~~~~~

These receive the information required by the LearningMechanism's `function <LearningMechanism_Function>`:

.. _LearningMechanism_Activation_Input:

* `ACTIVATION_INPUT`
   This receives the value of the input to the MappingProjection being learned (that is, the
   value of its `sender <MappingProjection.sender>`).  It's value is assigned as the first item of the
   LearningMechanism's `variable <LearningMechanism.variable>` attribute.

.. _LearningMechanism_Activation_Output:

* `ACTIVATION_OUTPUT`
   This receives the value of the LearningMechanism's `error_source <LearningMechanism_Additional_Attributes>`
   (that is, the output of the ProcessingMechanism to which the MappingProjection being learned projects).  By
   default, this uses the `primary outputState <OutputState_Primary>` of the `error_source`, but a different
   outputState can be designated in the `parameter dictionary <ParameterState_Specifying_Parameters>` for the
   params argument of the `error_source`, by including an entry with `MONITOR_FOR_LEARNING` as its key and a list
   containing the desired outputState(s) as its value. The value of the `ACTIVATION_OUTPUT` inputState is assigned
   as the second item of the LearningMechanism's `variable <LearningMechanism.variable>` attribute.

.. _LearningMechanism_Input_Error_Signal:

* `ERROR_SIGNAL`
   This receives an error_signal from either an `ObjectiveMechanism` or another LearningMechanism.
   If the MappingProjection being learned projects to the `TERMINAL` mechanism of the process or system being learned,
   or is not part of a `multilayer learning sequence <LearningMechanism_Multi_Layer>`, then the error_signal comes
   from an ObjectiveMechanism. If the MappingProjection is part of a multilayer learning sequence, then the
   LearningMechanism receives the error_signal from the next LearningMechanism in the sequence (i.e., the layer "above"
   it).  Its value is assigned as the third item of the LearningMechanism's
   `variable <LearningMechanism.variable>` attribute.

   .. note::
      The value of a LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` inputState is distinct
      from its `error_signal` attribute. The former is received from an ObjectiveMechanism or another
      LearningMechanism.  The latter is generated as a result of the LearningMechanism's
      `function <LearningMechanism.function>` and potentially passed on to other LearningMechanisms.

.. _LearningMechanism_Function:

Learning Function
~~~~~~~~~~~~~~~~~

This uses the three values received by the LearningMechanism's `inputStates <LearningMechanism_InputStates>` to
calculate a `learning_signal` and its own `error_signal`.  The `learning_signal` is the set of changes to the
`matrix <MappingProjection.matrix>` parameter of the MappingProjection required to reduce the value received by the
LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` inputState .  In
`multilayer learning <LearningMechanism_Multi_Layer>`, the `error_signal` it calculates reflects the contribution --
to the error_signal received -- made by the input to the MappingProjection being learned and the current value of
its `matrix <MappingProjection.matrix>` parameter (i.e., before it has been modified). The default
`function <LearningMechanism.function>` is BackPropagation` (also known as the *Generalized Delta Rule*; see
`Rumelhart et al., 1986 <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_).  However, it can be any
other PsyNeuLink `LearningFunction`
COMMENT:
, or any other python function that takes as its input a value with three 1d arrays or
lists, and returns two 1d arrays or lists.  The two return values
COMMENT
It returns two values that
are assigned to the LearningMechanism's
`learning_signal` and `error_signal` attributes, respectively, as well as to its two outputStates, as described below.

.. _LearningMechanism_OutputStates:

OutputStates
~~~~~~~~~~~~

These receive the output of the LearningMechanism's `function <LearningMechanism.function>`:

.. _LearningMechanism_Learning_Signal:

* `LEARNING_SIGNAL`
   This is assigned the value used to modify the `matrix <MappingProjection.matrix>` parameter
   of the MappingProjection being learned.  It is assigned as the `sender <LearningProjection.sender>` for the
   LearningProjection that projects to the MappingProjection.  It's value is accessible as the LearningMechanism's
   `learning_signal` attribute, and as the first item of the LearningMechanism's
   `outputValue <LearningMechanism.outputValue>` attribute.

.. _LearningMechanism_Output_Error_Signal:

* `ERROR_SIGNAL`
   This receives the error_signal used to calculate the learning_signal, which may have been
   weighted by the contribution that the MappingProjection and the mechanism to which it projects made to the
   `error_signal` received by the LearningProjection.  If the LearningMechanism is in a
   `multilayer learning sequence <LearningMechanism_Multi_Layer>`, it serves as the `sender <MappingProjection.sender>`
   for a MappingProjection to the LearningMechanism for the MappingProjection before it in the sequence (i.e.,
   the layer "below" it).  It's value is accessible as the LearningMechanism's `learning_signal` attribute,
   and as the first item of the LearningMechanism's `outputValue <LearningMechanism.outputValue>` attribute.

.. _LearningMechanism_Additional_Attributes:

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

In addition to these constituent components, a LearningMechanism is assigned attributes that refer to the
components being learned:

* `learned_projection`
   the MappingProjection for which the LearningMechanism is responsible;  that is, the one with the
   `matrix <MappingProjection.matrix>` parameter that the LearningMechanism modifies.
..
* `error_source`
   the mechanism that receives the `learned_projection`;  that is, the one that generates the output
   used to calculate the error_signal that the LearningMechanism attempts to reduce.

.. _LearningMechanism_Learning_Rate:

* `learning_rate <LearningMechanism.learning_rate>`
   the learning rate for the LearningMechanism (used to specify the :keyword:`learning_rate` parameter for its
   `function <LearningMechanism.function>`.  Specifiying this (or the learning_rate parameter of the
   `function <LearningMechanism.function>` directly) supercedes any specification of a learning_rate for
   any `process <Process.Process_Base.learning_rate>` and/or `system <System.System_Base.learning_rate>` to which
   the LearningMechanism belongs.  The default is `None`, in which case the LearingMechanism (and its
   `function <LearningMechanism.function>`) inherit the specification of the `learning_rate
   <Process.Process_Base.learning_rate>` for the process in which the LearningMechanism is being executed.
   If that is `None`, then it inherits it from the system in which it is being executed.  If that is also `None`,
   then it uses the default value assigned by its `function <LearningMechanism.function>`.

COMMENT:
@@@ THE FOLLOWING SECTION SHOULD BE MOVED TO THE "USER'S MANUAL" WHEN THAT IS WRITTEN
COMMENT
.. _LearningMechanism_Learning_Configurations:

Learning Configurations
~~~~~~~~~~~~~~~~~~~~~~~

When learning is specified for a `MappingProjection`, a `process <Process_Learning>`, or a
`system <System_Execution_Learning>`, PsyNeuLink automatically creates all of the components required for the
`MappingProjections <MappingProjection>` between `ProcessingMechanisms <ProcessingMechanism>` in that composition to
be learned.  The type of components that are generated depends on the :ref:`learning function <LearningFunction>`
specified, and the configuration of the composition.  All of the learning components of a system can be displayed
using the system's `show_graph` method with its **show_learning** argument assigned :keyword:`True`.

.. _LearningMechanism_Single_Layer:

Single layer learning
^^^^^^^^^^^^^^^^^^^^^

This is the case when only a single MappingProjection is specified for learning, or the LearningMechanism's function
only considers the output of its `error_source <LearningMechanism_Additional_Attributes>`  when computing the changes
that will be made to the `learned_projection's  <LearningMechanism_Additional_Attributes>`
`matrix <MappingProjection.matrix>` (e.g., `Reinforcement`).  In this case, a single `ObjectiveMechanism` and
LearningMechanism are created for the learned_projection <LearningMechanism_Additional_Attributes>`, if they do not
already exist, along with the following MappingProjections:

* from an outputState of the LearningMechanism's `error_source <LearningMechanism_Additional_Attributes>` to the
  ObjectiveMechanism's `SAMPLE` :ref:`inputState <LINK>`.  By default, the
  `primary outputState <OutputState_Primary>` of the error_souce is used;
  however, this can be modified by specifying its `MONITOR_FOR_LEARNING` parameter
  (see ` above <LearningMechanism_Activation_Output>`).

* from the process or system to the ObjectiveMechanism's `TARGET` :ref:`inputState <LINK>`;

* from the ObjectiveMechanism's `primary outputState <OutputState_Primary>` to the LearningMechanism's
  `ERROR_SIGNAL <LearningMechanism_Activation_Input>` inputState .

In addition, a `LearningProjection` is created from the LearningMechanism's
`LEARNING_SIGNAL <LearningMechanism_Learning_Signal>` outputState to the `matrix` `parameterState <ParameterState>`
for the `learned_projection <LearningMechanism_Additional_Attributes>`.  Because this case involves only a single
layer of learning, *no* projection is created or assigned to the LearningMechanism's
`ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` outputState.

.. _LearningMechanism_Single_Layer_Learning_Figure:

    **Components for Single Layer Learning**

    .. figure:: _static/LearningMechanism_Single_Layer_Learning_fig.pdf
       :alt: Schematic of mechanisms and projections involved in learning for a single MappingProjection
       :scale: 50%

       ObjectiveMechanism, LearningMechanism and associated projections created for a single learned_projection
       and error_source.  Each mechanism is labeled by its type (uppler line, in bold) and its designated
       status in the process and/or system to which it belongs (lower line, caps).  Italicized labels beside a
       component indicates the attribute of the LearningMechanism with which it is associated.


.. _LearningMechanism_Multi_Layer:

Multilayer learning
^^^^^^^^^^^^^^^^^^^

This is the case when a set of MappingProjections are being learned that are in a sequence (such as the `pathway` of a
process); that is, in which each projects to a ProcessingMechanism that is the `sender <MappingProjection.sender>` for
the next MappingProjection in the sequence (see the `figure <LearningMechanism_Multilayer_Learning_Figure>` below).
This requires the use of a learning function that can calculate the influence that each MappingProjection and its input
have on the error that the LearningMechanism receives from the next one in the sequence (e.g., `BackPropagation`).
In multilayer learning, the components created depend on the position of the
`learned_projection <LearningMechanism_Additional_Attributes>` and
`error_source <LearningMechanism_Additional_Attributes>` in the sequence.  If these are the last ones in the
sequence, they are treated in the same way as `single layer learning <LearningMechanism_Single_Layer>`.  This is the
case if the `error_source` is a standalone mechanism (one not in a process or system), the `TERMINAL` mechanism of a
standalone process (i.e., one not in a system), or the `TERMINAL` of all of the processes to which it belongs in a
system (and therefore a `TERMINAL` for the system).  In these cases, as for single layer learning,
an `ObjectiveMechanism` is created that receives the output of the `error_source` as well as the target for learning
(see `LearningMechanisms_Targets` below), and projects to a LearningMechanism that is created for the
`learned_projection`.  For all others, the following MappingProjections are created (shown in the `figure
<LearningMechanism_Multilayer_Learning_Figure>` below):

* from the `sender <MappingProjection.sender` of the `learned_projection` to the LearningMechanism's
  `ACTIVATION_INPUT` `inputState <LearningMechanism_Activation_Input>`.
..
* from the `error_source` to the LearningMechanism's
  `ACTIVATION_OUTPUT` `inputState <LearningMechanism_Activation_Output>`.
..
* from the `ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` outputState of the LearningMechanism for the
  next MappingProjection in the sequence (i.e., the layer "above" it) to the LearningMechanism's
  `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` inputState.

In addition, a `LearningProjection` is created from each LearningMechanism's
`LEARNING_SIGNAL <LearningMechanism_Learning_Signal>` outputState to the `matrix` `parameterState <ParameterState>`
of its `learned_projection`.  If the `learned_projection` is the first in the sequence, then *no* projection is
created or assigned to its LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Output_Error_Signal>` outputState.

.. _LearningMechanism_Multilayer_Learning_Figure:

    **Components for Multilayer Learning**

    .. figure:: _static/LearningMechanism_Multilayer_Learning_fig.pdf
       :alt: Schematic of mechanisms and projections involved in learning for a sequence of MappingProjections
       :scale: 50%

       ObjectiveMechanism and LearningMechanisms and associated projections created for a sequence of two
       MappingProjections specified for learning.  Each mechanism is labeled by its type (uppler line, in bold) and
       its designated status in the process and/or system to which it belongs (lower line, caps).  Italicized labels
       beside a component indicates the attribute of the LearningMechanism with which it is associated.

.. _LearningMechanism_Targets:

**TARGET mechanisms**: receive the targets specified for learning.  When learning is specified for a `process
<Process_Learning>` or `system <System_Execution_Learning>`, the `ObjectiveMechanism`  that will receive its
`targets <Run_Targets>` (specified in the call to its :keyword:`execute` or :keyword:`run` method) are identified and
designated as `TARGET` mechanisms. These are listed in the process` or system's :keyword:`targetMechanisms` attribute.
It is important to note that the status of a `ProcessingMechanism` in a system takes precedence over its status in any
of the processes to which it belongs. This means that even if a mechanism is the `TERMINAL` of a particular process,
if that process is combined with others in a system, the mechanism appears in any of those other processes,
and it is not the `TERMINAL` of all of them, then it will *not* be the `TERMINAL for the system.  As consequence,
although it will project to a `TARGET` mechanism in the process for which it is the `TERMINAL`, it will not do so in
the system (see :ref:`figure below <LearningProjection_Target_vs_Terminal_Figure>` for an example).  Finally, if a
mechanisms is the `TERMINAL` for more than one process used to create a system (that is, the pathways for those
processes converge on that mechanism), only one ObjectiveMechanism will be created for it in the system.

.. _LearningProjection_Target_vs_Terminal_Figure:

    **TERMINAL** and **TARGET** Mechanisms in Learning

    .. figure:: _static/LearningMechanism_TERMINAL_vs_TARGET_fig.pdf
       :alt: Schematic of mechanisms and projections involved in learning
       :scale: 50 %

       Mechanism 3 is the `TERMINAL` mechanism for Process A, However, it is also an `INTERNAL` mechanism of Process B.
       Therefore, Mechanism 3 is designated as an `INTERNAL` mechanism for the system, and Mechanism 4 is its `TERMINAL`
       mechanism. As a consequence, if `BackPropagation` is used for learning, then Mechanism 4 is an
       `ObjectiveMechanism` and designated as a `TARGET`, while Mechanism 3 is a LearningMechanism
       and designated as a `MONITORING` mechanism.

.. _LearningMechanism_Execution:

Execution
---------

LearningMechanisms are executed after all of the ProcessingMechanisms in the process or system to which it belongs have
been executed, including the ObjectiveMechanism(s) that provide the `error_signal` to each LearningMechanism.  When the
LearningMechanism is executed, it uses the value of its `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>`
inputState to calculate changes to the `matrix <MappingProjection.MappingProjection.matrix>` of its
`MappingProjection`.  The changes are assigned as the value of its `learning_signal` attribute (as well as the 1st item
of its `outputValue <LearningMechanism.outputValue>` attribute) and used as the `value <LearningProjection.value>` of
the `LearningProjection` from the LearningMechanism to the `MATRIX` parameterState of its `learned_projection`.
However, these but are not applied to the `matrix <MappingProjection.MappingProjection.matrix>` itself until the next
time the `learned_projection` is executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).
In addition, the LearningMechanism assigns the `error_signal` signal computed by its
`function <LearningMechanism.function>` to its `error_signal` attribute (as well as the 2nd item of its
`outputValue <LearningMechanism.outputValue>` attribute).

.. _LearningMechanism_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import _objective_mechanism_role
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.Functions.Function import BackPropagation, Logistic

# Params:

parameter_keywords.update({LEARNING_PROJECTION})
projection_keywords.update({LEARNING_PROJECTION})

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return :keyword:`true` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
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

# Used to name inputStates:
ACTIVATION_INPUT = 'activation_input'     # inputState
ACTIVATION_OUTPUT = 'activation_output'   # inputState
ERROR_SIGNAL = 'error_signal'             # inputState and outputState
LEARNING_SIGNAL = 'learning_signal'       #                outputState

input_state_names =  [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]
output_state_names = [LEARNING_SIGNAL, ERROR_SIGNAL]

ERROR_SOURCE = 'error_source'

DefaultTrainingMechanism = ObjectiveMechanism

class LearningMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningMechanism(AdaptiveMechanism_Base):
    """
    LearningMechanism(                       \
                 variable,                   \
                 error_source                \
                 function=BackPropagation    \
                 params=None,                \
                 name=None,                  \
                 prefs=None)

    Implements a mechanism that modifies the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.

    COMMENT:
        Description:
            LearningMechanism is a subtype of the AdaptiveMechanism Type of the Mechanism Category of Component
            It implements a mechanism that calculates changes to a projection's parameters.
            It's function takes the output of an ObjectiveMechanism (self.variable) and generates a
            learning_signal (2d arry of parameter changes) to be used by the recipient of a LearningProjection
            that projects from the LearningMechanism to a MappingProjection.

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
            + paramNames (dict)
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
        specifies a template for the three items required by the `function <LearningMechanism.function>: the input
        to the `learned_projection`, the output of the `error_source`, and the error_signal received by the
        LearningMechanism (see `variable <LearningMechanism.variable>` for details).

    error_source : ProcessingMechanism
        specifies the mechanism that generates the output on which the error_signal received by the LearningMechanism
        (in its `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` inputState) is based.

    function : LearningFunction or function
        specifies the function used to compute the `learning_signal` used by a LearningProjection, and the
        and `error_signal` passed to the next LearningMechanism in a
        `learning sequence <LearningMechanism_Learning_Configurations>`
        (see `function <LearningMechanism.function>` for details).

    learning_rate : float
        specifies the learning rate for this LearningMechanism (see `learning_rate <LearningMechanism.learning_rate>`
        for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
        projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
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

    learned_projection : MappingProjection
        the projection, the `matrix <MappingProjection.matrix>` of which is  modified by the LearningMechanism.

    COMMENT:
      error_output : 1d np.array
            the output of the next mechanism in the pathway (the one to which the `error_signal` pertains, and projected
            to by the mechanism that receives the projection being learned). Typically this comes from  the
            `LearningMechanism` for that next mechanism.  However, if the current LearningMechanism is for the last
            mechanism in a sequence of mechanisms being learned (often, but not necessarily a `TERMINAL` mechanism),
            then error_output is set to an array of 1's with a length equal to the length of the `error_signal`.

        error_source : ObjectiveMechanism or LearningMechanism
            the mechanism from which the LearningMechanism gets its `error_signal`.  The LearningMechanism receives a
            projection from the `error_source` to its `ERROR_SIGNAL <LearningMechanism.inputStates>` inputState.
            If the `error_source` is an ObjectiveMechanism, the projection is from its
            `primary outputState <OutputState_Primary>`.  If the `error_source` is another LearningMechanism,
            the projection is from its `ERROR_SIGNAL <LearningMechanism.outputStates>` outputState.  In either case,
            the MappingProjection uses an `IDENTITY_MATRIX`, and so the value of the outputState used for the
            `error_source` must be equal in length to the value of the LearningMechanism's `ERROR_SIGNAL` inputstate.
    COMMENT

    error_source : ProcessingMechanism
        the mechanism that generates the output upon which the error signal received by LearningMechanism
        (in its `ERROR_SIGNAL <LearningMechanism_Input_Error_SIgnal>` inputState) is based.

    function : LearningFunction or function : default BackPropagation
        specifies function used to compute the `learning_signal`.  Must take the following arguments:
        `input` (list or 1d array), `output` (list or 1d array), `derivative` (function) and `error` (list or 1d array).

    learning_rate : float : None
        determines the learning rate for the LearningMechanism.  It is used to specify the :keyword:`learning_rate`
        parameter for the LearningMechanism's `learning function <LearningMechanism.function>`
        (see description of `learning_rate <LearningMechanism_Learning_Rate>` above for additional details).

    error_signal : 1d np.array
        the error signal returned by the LearningMechanism's `function <LearningMechanism.function>`.  For
        `single layer learning <LearningMechanism_Single_Layer>`, this is the same as the value received in the
        LearningMechanism's `ERROR_SIGNAL <LearningMechanism_Input_Error_Signal>` inputState;  for `multilayer
        learning <LearningMechanism_Multi_Layer>`, it is a modified version of the value received, that takes
        account of the contribution to the error signal received, made by the learned_projection and its input.

    learning_signal : 2d np.array
        matrix of changes to be used by recipient of `LearningProjection` to adjust its parameters (e.g.,
        matrix weights, in rows correspond to sender, columns to receiver); same as `value <LearningMechanism.value>`.

    outputValue : 2d np.array
        the 1st item is the same as `learning_signal`;  the 2nd item is the same as `error_signal`.

    name : str : default LearningProjection-<index>
        the name of the LearningMechanism.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for projection.
        Specified in the `prefs` argument of the constructor for the projection;
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
        INPUT_STATES:input_state_names,
        OUTPUT_STATES:[{NAME:LEARNING_SIGNAL,
                        INDEX:0},
                       {NAME:ERROR_SIGNAL,
                        INDEX:1}]
    })

    @tc.typecheck
    def __init__(self,
                 variable:tc.any(list, np.ndarray),
                 error_source:tc.optional(Mechanism)=None,
                 function:is_function_type=BackPropagation,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
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
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)
        TEST = True

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
        """Validate error_source as an Objective mechanism or another LearningMechanism
        """

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

        try:
            if not isinstance(target_set[ERROR_SOURCE], (ObjectiveMechanism, LearningMechanism)):
                raise LearningMechanismError("{} arg for {} must be an ObjectiveMechanism or another LearningMechanism".
                                             format(ERROR_SOURCE, self.name))

        except KeyError:
            pass

    def _instantiate_attributes_before_function(self, context=None):
        """Instantiates MappingProjection from error_source (if specified) to the LearningMechanism

        Also assigns learned_projection attribute (to MappingProjection being learned)
        """

        super()._instantiate_attributes_before_function(context=context)


        if self.error_source:
            _instantiate_error_signal_projection(sender=self.error_source, receiver=self)

    def _instantiate_function(self, context=None):
        super()._instantiate_function(context=context)


    # MODIFIED 3/22/17 NEW:
    def _instantiate_attributes_after_function(self, context=None):

        super()._instantiate_attributes_after_function(context=context)

        if self._learning_rate is not None:
            self.learning_rate = self._learning_rate
    # MODIFIED 3/22/17 END


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

        # # TEST PRINT:
        # print("\n@@@ EXECUTED: {}".format(self.name))

        self.value = [self.learning_signal, self.error_signal]
        return self.value

    # IMPLEMENTATION NOTE: Assumes that the LearningMechanism projects to and modifies only a single MappingProjection
    @property
    def learned_projection(self):
        learning_projections = self.outputStates[LEARNING_SIGNAL].sendsToProjections
        if learning_projections:
            return learning_projections[0].receiver.owner
        else:
            return None

    @property
    def learning_rate(self):
        return self.function_object.learning_rate

    @learning_rate.setter
    def learning_rate(self, assignment):
        self.function_object.learning_rate = assignment


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _instantiate_error_signal_projection(sender, receiver):
    """Instantiate a MappingProjection to carry an error_signal to a LearningMechanism

    Can take as the sender an `ObjectiveMechanism` or a `LearningMechanism`.
    If the sender is an ObjectiveMechanism, uses its `primary outputState <OutputState_Primary>`.
    If the sender is a LearningMechanism, uses its `ERROR_SIGNAL <LearningMechanism.outputStates>` outputState.
    The receiver must be a LearningMechanism; its `ERROR_SIGNAL <LearningMechanism.inputStates>` inputState is used.
    Uses and IDENTITY_MATRIX for the MappingProjection, so requires that the sender be the same length as the receiver.

    """

    if isinstance(sender, ObjectiveMechanism):
        sender = sender.outputState
    elif isinstance(sender, LearningMechanism):
        sender = sender.outputStates[ERROR_SIGNAL]
    else:
        raise LearningMechanismError("Sender of the error signal projection {} must be either "
                                     "an ObjectiveMechanism or a LearningMechanism".
                                     format(sender))

    if isinstance(receiver, LearningMechanism):
        receiver = receiver.inputStates[ERROR_SIGNAL]
    else:
        raise LearningMechanismError("Receiver of the error signal projection {} must be a LearningMechanism".
                                     format(receiver))

    if len(sender.value) != len(receiver.value):
        raise LearningMechanismError("The length of the outputState ({}) for the sender ({}) of "
                                     "the error signal projection does not match "
                                     "the length of the inputState ({}) for the receiver ({})".
                                     format(len(sender.value), sender.owner.name,
                                            len(receiver.value),receiver.owner.name))

    return MappingProjection(sender=sender,
                             receiver=receiver,
                             matrix=IDENTITY_MATRIX,
                             name = sender.owner.name + ' ' + ERROR_SIGNAL)
