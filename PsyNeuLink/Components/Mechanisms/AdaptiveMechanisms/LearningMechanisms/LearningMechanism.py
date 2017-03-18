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

A LearningMechanism is an `AdaptiveMechanism` that modifies a parameter (usually the `matrix <MappingProjection.matrix>`
of a `MappingProjection`.  It's function takes an `error_signal` (usually the output of an `ObjectiveMechanism` or
another `LearningMechanism`) and generates a `learning_signal` that is conveyed to a MappingProjection by way of a
`LearningProjection`.  A LearningMechanism can modify only MappingProjections that link ProcessingMechanisms in the
same `system <System>` or `process <Process>` to which it belongs, and is executed after all of the
ProcessingMechanisms in that system or process have been executed.  If it belongs to a system, it is executed before
the `ControlMechanism` for that system has been executed.

@@ DEFINE LEARNING SEQUENCE

@@@ SEARCH FOR LearningProjection_Automatic_Creation AND REPLACE WITH REFERENCE TO THIS LABEL:
.. _LearningMechanism_Creation:

Creating a LearningMechanism
----------------------------

LearningMechanisms can be created in any of the ways that can be used to `create mechanisms <Mechanism_Creation>`.
More commonly, however, they are created automatically in the following cases:

* when the learning attribute is specified for a :ref:`system <LINK>` or :ref:`process <LINK>`;

* when a `LearningProjection` (or the keyword `LEARNING`) is specified in the
  `tuple that specifies a MappingProjection <MappingProjection_Tuple_Specification>` in the `pathway` of a process;

* whenever a `LearningProjection` is created without specifying its `sender <LearningProjection.sender>` attribute.

In these instances, an `ObjectiveMechanism`, LearningProjections <LearningProjection>`, and any additional projections
required to implement learning that do not already exist are also instantiated.  These components are shown in the
:ref:`figure <LearningMechanism_Simple_Learning_Figure>` below, and described under
:ref:`Structure <LearningProjection_Structure>`).  If learning is specified for a process or system, these
are generated for each MappingProjection that will be modified by learning (e.g., for a process, this includes the
MappingProjections between each of the mechanisms in the `pathway` of the process).


.. _LearningMechanism_Simple_Learning_Figure:

    **Components of Learning**

    .. figure:: _static/LearningProjection_Simple_Learning_fig.jpg
       :alt: Schematic of mechanisms and projections involved in learning
       :scale: 50%

       Learning mechanisms (darker background) and associated projections created for a set of mechanisms specified for
       learning (lighter backgrounds).  Each mechanism is labeled by its type (uppler line, in bold) and its designated
       status in the process and/or system to which it belongs (lower line, caps).  Italicized labels beside each
       mechanism indicate the attributes of the LearningProjection with which they are associated.

.. _LearningProjection_Structure:


COMMENT:
    The `receiver <MappingProjection.MappingProjection.receiver>` for each of those MappingProjections is assigned as the
    `errorSource <LearningProjection.errorSource>` for the LearningProjection.  Each errorSource must project to a
    :doc:`MonitoringMechanism`, which is assigned as the the LearningSignal's sender, and provides it with an
    `error_signal`. If the `errorSource` assigned to a LearningProjection already has a projection to a
    MonitoringMechanism, then that mechanism is simply assigned as the LearningProjection's
    `sender <LearningProjection.sender>`; if the `errorSource` does not project to any MonitoringMechanism, then one is
    created for it at the same time that the LearningProjection is created.

    The type of MonitoringMechanism created depends on the type of learning. For `Reinforcement Learning <Reinforcement>`,
    a `ComparatorMechanism` is created, and given a MappingProjection from the `errorSource`. For
    `BackPropagation`, the type of  MonitoringMechanism created also depends on the `errorSource` itself.  If
    the `errorSource` provides the output that will be compared with the target stimulus then, as for Reinforcement
    Learning, a ComparatorMechanism is created.  This is the case if the `errorSource` is a standalone
    mechanism (one not in a process or system), the `TERMINAL` mechanism of a standalone process (i.e., one not
    in a system), or the `TERMINAL` mechanism of a system.  However, if the `errorSource` lies deeper in a process
    or system -- that is, if it is an `ORIGIN` or `INTERNAL` mechanism -- then a `WeightedErrorMechanism` mechanism
    is created.  This gets its error information from the MonitoringMechanism for the `errorSource` "after" it in the
    process or system (i.e., the one to which it projects, and that is one closer to
    the target).  Therefore, a MappingProjection is created that projects to it from that next `errorSource`.
COMMENT

Structure
---------

A LearningMechanism has three `inputStates <InputState>`, a learning `function <LearningMechanism.function>`,
and two `outputStates <OutputStates>` that are used to receive, compute, and transmit the information need to modify
the MappingProjection for which it is responsible.

.. _LearningMechanism_InputStates:

InputStates
~~~~~~~~~~~

These receive the information required by the LearningMechanism's `function <LearningMechanism.function>` to compute the
`learning_signal` used to modify the matrix parameter of the MappingProjection for which it is responsible:

* `ACTIVATION_INPUT` - this receives the value of the input to the MappingProjection being learned (that is, the
   value of its `sender <MappingProjection.sender>`).

* `ACTIVATION_OUTPUT` - this receives the value of the output of the ProcessingMechanism to which the MappingProjection
   being learned projects (that is, the value of its `receiver <MappingProjection.receiver>`.  By default, this is the
   value of the receiver's `primary outputState <OutputState_Primary>`, but a different outputState can be designated
   in a `parameter dictionary <ParameterState_Specifying_Parameters>` of the receiver's params argument, by including
   an entry with `MONITOR_FOR_LEARNING` as its key and a list containing the desired outputState(s) as its value.

* `ERROR_SIGNAL` - this receives an `error_signal` from either an `ObjectiveMechanism` or another LearningMechanism.
  If the MappingProjection being learned projects to the `TERMINAL` mechanism of the process or system being trained,
  or is not part of a :ref:`learning sequence <LINK>`, then the `error_signal` must come from an ObjectiveMechanism.
  If the MappingProjection is part of a learning sequence, then it receives its `error_signal` from the next
  LearningMechanism in the sequence.


.. _LearningMechanism_Function:

Learning Function
~~~~~~~~~~~~~~~~~


.. _LearningMechanism_OutputStates:

OutputStates
~~~~~~~~~~~

These receive the output of the LearningMechanism's `function <LearningMechanism.function>`:

* `LEARNING_SIGNAL` - this receives the value used to modify the `matrix <MappingProjection.matrix>` parameter
  of the MappingProjection being learned.  It is assigned as the `sender <LearningProjection.sender>` for the
  LearningProjection that projects to the MappingProjection.

* `ERROR_SIGNAL` - this receives the error_signal used to calculate the learning_signal, which may have been
  weighted by the contribution that the MappingProjection and the mechanism to which it projects made to the
  `error_signal` received by the LearningProjection.  If the LearningMechanism is in a learning sequence,
  it serves as the `sender <MappingProjection.sender>` for a MappingProjection to the next
  LearningMechanism in the sequence.






The following components and information are required by a LearningMechanism
(see :ref:`figure above <LearningMechanism_Simple_Learning_Figure>`):

**MappingProjection**: the projection that is modified by the LearningMechanism.  The LearningMechanism sends a
`LearningProjection` to the `parameterState <ParameterState>` for the `matrix <MappingProjection.matrix>` parameter
of that MappingProjection.

**Error source**: the ProcessingMechanism to which the `MappingProjection` that is being learned projects;  it is the
mechanism responsible for the component of the error that the LearningMechanism tries to reduce.  It is
referenced by the LearningMechanism's `error_source` attribute.  The `error_source` must project to either an
`ObjectiveMechanism` or another LearningMechanism (see below). By default, the projection from the error_source
comes from its `primary outputState <OutputState_Primary>`, but a different outputState can be specified by
including an entry with `MONITOR_FOR_LEARNING` as its key in a
`parameter dictionary <ParameterState_Specifying_Parameters>` for the `error_source`, and assigning it a list with
the desired outputState(s) as its value.  When a LearningMechanism is `created automatically, if its `error_source`
already has a projection to an ObjectiveMechanism or anotherLearningMechanism, then that is used; if its
`error_source` does not project to one of these, then one of the appropriate type is created (see below)
and assigned a MappingProjection from the `error_source`.

.. _LearningProjection_MonitoringMechanism:

**Error signal**: this is the value that the LearningMechanism seeks to reduce.  It is
Usually it comes from an
`ObjectiveMechanism or another LearningMechanism**:  This calculates the `error_signal` used by the
current
LearningMechanism to reduce the contribution of its `error_source` to the error.  Which of these is required, and how it
calculates the `error_signal`, depend on the `function <LearningProjection.function>` that the LearningMechanism uses
for learning. For `Reinforcement`, an `ObjectiveMechanism` is always used. This receives a MappingProjection directly
from the `error_source <LearningProjection.errorSource>`, and receives a **target** stimulus from the process or
system to

xxxxxx

which the `errorSource <LearningProjection.errorSource>` belongs.  It calculates the `error_signal` by comparing the
output of the `errorSource <LearningProjection.errorSource>` with the target stimulus provided when the process or
system is `run <Run_Targets>`. For `BackPropagation`, the type of MonitoringMechanism depends on the
`errorSource <LearningProjection.errorSource>`. If the `errorSource <LearningProjection.errorSource>` receives a target
directly, then a `ComparatorMechanism` is used.  This is the case if the `errorSource <LearningProjection.errorSource>`
is a standalone mechanism (one not in a process or system), the `TERMINAL` mechanism of a standalone
process (i.e., one not in a system), the `TERMINAL` mechanism of a system, or it has been specified explicitly as a
`TARGET <LINK>` mechanism.  However, if the `errorSource <LearningProjection.errorSource>` lies deeper in any process
to which it  belongs (i.e., it is not a `TERMINAL` mechanism), and has not been explicitly specified as a
`TARGET <LINK>` and therefore does not receive a target directly, then a `WeightedErrorMechanism` mechanism is used.
This receives a MappingProjection carrying its error information from the MonitoringMechanism for the
`errorSource <LearningProjection.errorSource>` "after" it in the process (i.e., the one to which it projects, and that
is one closer to the target), rather than from a target stimulus. It calculates its `error_signal` by taking account of
the contribution that its `errorSource <LearningProjection.errorSource>` makes to the `error_signal` of the *next*
mechanism in the process or system.

**LearningProjection**:  this calculates the changes to the `matrix <MappingProjection.MappingProjection.matrix>`
parameter of the MappingProjection to which the LearningProjection projects (i.e., the owner of its
`receiver <LearningProjection.receiver>`), so as to reduce the error generated by its
`errorSource <LearningProjection.errorSource>`. It uses the `error_signal` received from the `MonitoringMechanism` (i.e.,
the owner of its `sender <LearningProjection.sender>`), to which the `errorSource <LearningProjection.errorSource>`
projects. The weight changes it provides to its `receiver <LearningProjection.receiver>`are stored in its
`weightChangeMatrix` attribute.

.. _LearningProjection_Targets:

**TARGET mechanisms**: receive the targets specified for learning.  When learning is specified for a `process
<Process_Learning>` or `system <System_Execution_Learning>`, the `MonitoringMechanism(s) <MonitoringMechanism>`  that
will receive its `targets <Run_Targets>` (specified in the call to its :keyword:`execute` or :keyword:`run` method)
are identified and designated as `TARGET` mechanisms. These are listed in the process` or system's
:keyword:`targetMechanisms` attribute. All other MonitoringMechanism(s) are designated as `MONITORING`. `TARGET`
mechanisms must be `ComparatorMechanisms <ComparatorMechanism>`; if the `BackPropagation` function is used for
learning, they must also be associated with (i.e., receive a projection from) a `TERMINAL` mechanism.  It is
important to note that, for this purpose, the status of a mechanism in a system takes precedence over its status in
any of the processes to which it belongs. This means that, although all of the `TERMINAL` mechanisms of a system are
associated with `TARGET` mechanisms, this *not* necessarily true for the `TERMINAL` mechanism of a process.  This is
because a mechanism may be the `TERMINAL` mechanism of a process, but not of the system to which it belongs
(see :ref:`figure below <_LearningProjection_Target_vs_Terminal_Figure>` for an example).  In such cases, the mechanism
is assigned a WeightedErrorMechanism rather than a ComparatorMechanism for learning, and is designated as a
`MONITORING` mechanism and *not* a `TARGET` mechanism.

.. _LearningProjection_Target_vs_Terminal_Figure:

    **TERMINAL** and **TARGET** Mechanisms in Learning

    .. figure:: _static/LearningProjection_TERMINAL_vs_TARGET_fig.jpg
       :alt: Schematic of mechanisms and projections involved in learning
       :scale: 50 %

       Mechanism 3 is the `TERMINAL` mechanism for Process A, However, it is also an `INTERNAL` mechanism of Process B.
       Therefore, Mechanism 3 is designated as an `INTERNAL` mechanism for the system, and Mechanism 4 is its `TERMINAL`
       mechanism. As a consequence, if `BackPropagation` is used for learning, then Mechanism 4 is assigned a
       `ComparatorMechanism` and designated as a `TARGET`, while Mechanism 3 is assigned a `WeightedErrorMechanism` and
       designated as a `MONITORING` mechanism.  This also means that, as long as Process A is specified as part of that
       system, it cannot be executed on its own with learning enabled (since it will have no `TARGET`)

.. _LearningProjection_Function:

**Function**:  calculates the changes to the `matrix <MappingProjection.MappingProjection.matrix>` parameter
of the LearningProjection's `mappingProjection required to reduce the error for its
`errorSource <LearningProjection.errorSource>`.  The result is assigned to the LearningProjection's
`weightChangeMatrix` attribute. The default `function <LearningProjection.function>` is BackPropagation` (also known
as the *Generalized Delta Rule*; see
`Rumelhart et al., 1986 <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_). However, it can be
assigned to other functions that implement different learning algorithms, as long as it is compatible with the
:keyword:`function` of the LearningProjection's `errorSource <LearningProjection.errorSource>` (how the `error_signal`
is computed depends on the nature of the function that generated the error); failure to match the `function
<LearningProjection.function>` for the LearningProjection with  the :keyword:`function` of its
`errorSource <LearningProjection.errorSource>`  will generate an error.

.. _LearningProjection_Execution:

Execution
---------

LearningMechanisms are executed after all of the ProcessingMechanisms in a process or system have executed,
including the ObjectiveMechanism(s) that provide the `error_signal` to each LearningMechanism.  When the
LearningMechanism is executed, it uses its `error_signal` to calculate changes to the
`matrix <MappingProjection.MappingProjection.matrix>` of its `MappingProjection`. Changes to the matrix are
calculated so as to reduce the `error_signal`. The changes are assigned as the `value <LearningProjection.value>` of
the `LearningProjection` from the LearningMechanism to the MappingProjection, but are not applied to
its `matrix <MappingProjection.MappingProjection.matrix>` until the next time the `MappingProjection` is executed
(see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

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

MECH_LEARNING_RATE = 'mech_learning_rate'

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
                 mech_learning_rate=None     \
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
        takes four items specifying: 1) the input to the `MappingProjection` being learned; 2) the resulting output of
        the mechanism to which it projects; the output of the next mechanism in the pathway; and the error signal
        from the next LearningMechanism in the pathway (see `variable <LearningMechanism.variable>` for details).

    COMMENT
        activation_derivative : Function or function
            specifies the derivative of the function of the mechanism that receives the `MappingProjection` being learned
            (see `activation_derivative` for details).
    COMMENT

    error_source : ObjectiveMechanism or LearningMechanism
        specifies the mechanism from which the LearningMechanism gets its `error_signal` (see `error_source` for
        details).

    function : LearningFunction or function
        specifies the function used to compute the `learning_signal` (see `function <LearningMechanism.function>` for
        details).

    mech_learning_rate : float
        specifies the learning rate for this LearningMechanism (see `mech_learning_rate` for details).

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
        has three items, each of which is a 1d np.array: `activation_input`, `activation_output`, and `error_signal`.

    activation_input : 1d np.array
        the input to the `MappingProjection` being learned.

    activation_output : 1d np.array
        the output of the mechanism that receives the `MappingProjection` being learned.

    COMMENT:
        activation_derivative : Function or function
            the derivative of the function of the mechanism that receives the `MappingProjection` being learned.
    COMMENT

    COMMENT:
        error_output : 1d np.array
            the output of the next mechanism in the pathway (the one to which the `error_signal` pertains, and projected
            to by the mechanism that receives the projection being learned). Typically this comes from  the
            `LearningMechanism` for that next mechanism.  However, if the current LearningMechanism is for the last
            mechanism in a sequence of mechanisms being learned (often, but not necessarily a `TERMINAL` mechanism),
            then error_output is set to an array of 1's with a length equal to the length of the `error_signal`.
    COMMENT

    error_signal : 1d np.array
        the error signal, typically generated by an `LearningMechanism` associated with the next mechanism in the
        learning sequence (i.e., the one projected to by the mechanism that receives the `MappingProjection` being
        learned).  If the LearningMechanism is for the last projection in a sequence being learned (often,
        but not necessarily the one that projects to the `TERMINAL` mechanism), or is an isolated (or only) projection
        being learned, then the `error_signal` comes from an `ObjectiveMechanism` that calculates the error from that
        receiver mechanism's output and a target input to the process being learned.

    error_source : ObjectiveMechanism or LearningMechanism
        the mechanism from which the LearningMechanism gets its `error_signal`.  The LearningMechanism receives a
        projection from the `error_source` to its `ERROR_SIGNAL inputState <LearningMechanism.inputStates>`.
        If the `error_source` is an ObjectiveMechanism, the projection is from its
        `primary outputState <OutputState_Primary>`.  If the `error_source` is another LearningMechanism,
        the projection is from its `ERROR_SIGNAL outputState <LearningMechanism.outputStates>`.  In either case,
        the MappingProjection uses an `IDENTITY_MATRIX`, and so the value of the outputState used for the
        `error_source` must be equal in length to the value of the LearningMechanism's `ERROR_SIGNAL` inputstate.

    COMMENT:
       MOVE THIS TO Backpropagation
        error_matrix : List or 2d np.array
            the matrix for the `MappingProjection` that projects *from* the mechanism that receives the MappingProjection
            being learned, *to* the next mechanism in the process or system, from which the `error_signal` was generated.
    COMMENT

    function : LearningFunction or function : default BackPropagation
        specifies function used to compute the `learning_signal`.  Must take the following arguments:
        `input` (list or 1d array), `output` (list or 1d array), `derivative` (function) and `error` (list or 1d array).

    mech_learning_rate : float : default 1.0
        determines the learning rate for the LearningMechanism.  It is used to specify the `learning_rate` parameter
        for the LearningMechanism's `learning function <LearningMechanism.function>`;  It is superceded by specification
        of a learning_rate for the `process <Process.Process_Base.learning_rate>` or
        `system <System.System_Base.learning_rate>` if either of those is specified.

    # objective_mechanism : Optional[ObjectiveMechanism or OutputState]
    #     the 'mechanism <Mechanism>` or its `outputState <OutputState>` that provides the `error_signal`
    #     used by the LearningMechanism's `function <LearningMechanism.function>` to compute the `learning_signal`.
    #     Typically this is an `ObjectiveMechanism`.

    learning_signal : 2d np.array
        matrix of changes to be used by recipient of `LearningProjection` to adjust its parameters (e.g.,
        matrix weights, in rows correspond to sender, columns to receiver); same as `value <LearningMechanism.value>`.

    value : 2d np.array
        same as `learning_signal`.

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
                 mech_learning_rate:float=1.0,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  mech_learning_rate=mech_learning_rate,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name
        # delete self.init_args[ERROR_MATRIX]
        # delete self.init_args[MECH_LEARNING_RATE]

        # # Flag for deferred initialization
        # self.value = DEFERRED_INITIALIZATION

        super().__init__(variable=variable,
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

        # self.activation_input = self.variable[ACTIVATION_INPUT_INDEX]
        # self.activation_output = self.variable[ACTIVATION_OUTPUT_INDEX]
        # self.error_signal = self.variable[ERROR_SIGNAL_INDEX]

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
        """

        super()._instantiate_attributes_before_function(context=context)

        if self.error_source:
            _instantiate_error_signal_projection(sender=self.error_source, receiver=self)


    def _instantiate_function(self, context=None):
        super()._instantiate_function(context=context)


    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute LearningMechanism function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # # MODIFIED 3/4/17 NEW:
        # # If error signal is from Objective function, make input = 1 so that when BP multiplies by it nothing happens
        # # if self.inputStates[ERROR_SIGNAL].receivesFromProjections:
        # if not INITIALIZING in context:
        #     if isinstance(self.error_source, ObjectiveMechanism):
        #         variable[ACTIVATION_INPUT_INDEX] = np.ones_like(variable[ACTIVATION_INPUT_INDEX])
        # # MODIFIED 3/4/17 END

        # COMPUTE LEARNING SIGNAL (dE/dW):
        self.learning_signal, self.error_signal = self.function(variable=variable, context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        # # TEST PRINT:
        # print("\n@@@ EXECUTED: {}".format(self.name))

        self.value = [self.learning_signal, self.error_signal]
        return self.value


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _instantiate_error_signal_projection(sender, receiver):
    """Instantiate a MappingProjection to carry an error_signal to a LearningMechanism

    Can take as the sender an `ObjectiveMechanism` or a `LearningMechanism`.
    If the sender is an ObjectiveMechanism, uses its `primary outputState <OutputState_Primary>`.
    If the sender is a LearningMechanism, uses its `ERROR_SIGNAL outputState <LearningMechanism.outputStates>`.
    The receiver must be a LearningMechanism; its `ERROR_SIGNAL inputState <LearningMechanism.inputStates>` is used.
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
                             matrix=IDENTITY_MATRIX)
