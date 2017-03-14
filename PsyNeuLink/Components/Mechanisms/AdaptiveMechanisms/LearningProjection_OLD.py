# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#                                              @@@@@@@ OLD @@@@@@
# *******************************************  LearningProjection ******************************************************

"""
.. _LearningProjection_Overview:

Overview
--------

A LearningProjection takes as its input the output of its `sender <LearningProjection.sender>` --
a `MonitoringMechanism <MonitoringMechanism>`.  It uses this as an `error_signal` to compute a `weightChangeMatrix`
that is assigned as the `value <LearningProjection.value>` of the LearningProjection.  This is used, in turn,
by the LearningProjection's `receiver <LearningProjection.receiver>` -- a `parameterState <ParameterState>` for
a `MappingProjection` to modify its  `matrix <MappingProjection.matrix>` parameter.  A LearningProjection can
be assigned different `functions <LearningProjection_Function>` to implement different learning algorithms,
which are associated with corresponding types of `MonitoringMechanisms <MonitoringMechanism>`.

.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

A LearningProjection can be created in any of the ways that can be used to `create a projection <Projection_Creation>`,
or by including it in the specification of a `system <System>`, `process <Process>`, or projection in the `pathway`
of a process.  Its `sender <LearningProjection.sender>` (the source of its `error_signal`) must be a
`MonitoringMechanism`, and its `receiver <LearningProjection.receiver>` must be the `parameterState <ParameterState>`
of a `MappingProjection`.  When a LearningProjection is created, its full initialization is
:ref:`deferred <Component_Deferred_Init>` until its `sender <LearningProjection.sender>` and
`receiver <LearningProjection.receiver>` have been fully specified.  This allows a LearningProjection to be created
before its `sender` and/or `receiver` have been created (e.g., before them in a script), by calling its constructor
without specifying its :keyword:`sender` or :keyword:`receiver` arguments.

It is also possible to create a LearningProjection without ever specifying its :keyword:`sender`.  In this case,
when it is initialized, the type of `MonitoringMechanism <MonitoringMechanism>` it requires will be inferred from
context, and created when it is initialized (see `Structure <LearningProjection_Structure>` below).  In contrast, a
LearningProjection's :keyword:`receiver` must be explicitly specified at some point.  Once that is done,
for the LearningProjection to be operational, initializaton must be completed by calling its `deferred_init`
method.  This is not necessary if learning has been specified for a `system <System_Execution_Learning>`,
`process <Process_Learning>`, or as the `projection <MappingProjection_Tuple_Specification>` in the `pathway` of a
process -- in those cases, deferred initialization is completed automatically.

COMMENT:
   REPLACE WITH THIS ONCE FUNCTIONALITY IS IMPLEMENTED
    Initialization will
    be  completed as soon as the LearningProjection has been assigned as the projection *from* a MonitoringMechanism (i.e.,
    it's ``sender`` has been specified) and as the projection *to* the parameterState of a MappingProjection (i.e.,
    its ``receiver`` has been specified).  This is handled automatically when the LearningProjection is specified as part of
    a system, process, or projection in the pathway of a process.
COMMENT

.. _LearningProjection_Automatic_Creation:

Automatic creation
~~~~~~~~~~~~~~~~~~

When learning is specified for a `process <Process_Learning>` or `system <System_Execution_Learning>`, or in a
`tuple that specifies a LearningProjection <MappingProjection_Tuple_Specification>`, PsyNeuLink automatically
generates a LearningProjection and the associated components required for learning to occur (shown in the
:ref:`figure <LearningProjection_Simple_Learning_Figure>` and described under
:ref:`Structure <LearningProjection_Structure>` below).  These are generated for each MappingProjection that will
be modified by learning;  for a process, this includes the MappingProjections between each of the mechanisms in the
process.

COMMENT:
    When learning is :ref:`specified for a process <Process_Learning>`, or in a
    :ref:`tuple that specifies a projection <LINK>`,  PsyNeuLink automatically generates the LearningProjection,
    MonitoringMechanisms, and corresponding projections required for learning to occur (shown in the
    :ref:`figure below <LearningProjection_Simple_Learning_Figure>`). More specifically, a LearningProjection
    is automatically created and  assigned to each MappingProjection for which it is specified (i.e., that it will
    modify). For a process, LearningProjections are created for the MappingProjection between each of the mechanisms
    in the process.

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

.. _LearningProjection_Simple_Learning_Figure:

    **Components of Learning**

    .. figure:: _static/LearningProjection_Simple_Learning_fig.jpg
       :alt: Schematic of mechanisms and projections involved in learning
       :scale: 50%

       Learning mechanisms (darker background) and associated projections created for a set of mechanisms specified for
       learning (lighter backgrounds).  Each mechanism is labeled by its type (uppler line, in bold) and its designated
       status in the process and/or system to which it belongs (lower line, caps).  Italicized labels beside each
       mechanism indicate the attributes of the LearningProjection with which they are associated.

.. _LearningProjection_Structure:

Structure
---------

The following components are required for learning
(see :ref:`figure above <LearningProjection_Simple_Learning_Figure>`):

**MappingProjection**: owner of the `parameterState <ParameterState>` to which the LearningProjection projects,
and of the `matrix <MappingProjection.MappingProjection.matrix>` to be modified by learning. It is referenced by the
LearningProjection's `mappingProjection` attribute.

**Error source**: ProcessingMechanism to which the `mappingProjection` that is being learned projects;  it is the
mechanism responsible for the component of the error that the LearningProjection tries to correct.  It is
referenced by the LearningProjection's `errorSource <LearningProjection.errorSource>` attribute.  The
`errorSource <LearningProjection.errorSource>` must project to a `MonitoringMechanism` (see below). By default,
the `primary outputState <OutputState_Primary>` of the `errorSource <LearningProjection.errorSource>` projects to the
MonitoringMechanism. However, a different outputState can be specified by including an
entry with `MONITOR_FOR_LEARNING` as its key in a `parameter dictionary <ParameterState_Specifying_Parameters>` for
the `errorSource <LearningProjection.errorSource>`, and assigning it a list with the desired outputState(s) as its
value. When a LearningProjection is `created automatically <LearningProjection_Automatic_Creation>`,
if its `errorSource <LearningProjection.errorSource>` already has a projection to a MonitoringMechanism,
then that one is used; if its `errorSource <LearningProjection.errorSource>` does not project to any
MonitoringMechanism, then one of an appropriate type is created (see below) and assigned a MappingProjection from the
`errorSource <LearningProjection.errorSource>`.

.. _LearningProjection_MonitoringMechanism:

**MonitoringMechanism**: its outputState serves as the `sender <LearningProjection.sender>` for the LearningProjection.
It calculates the `error_signal` used by the LearningProjection to reduce the contribution of its
`errorSource <LearningProjection.errorSource>` to the error.  The type of `MonitoringMechanism` required, and how it
calculates the `error_signal`, depend on the `function <LearningProjection.function>` that the LearningProjection uses
for learning. For `Reinforcement`, a `ComparatorMechanism` is used. This receives a MappingProjection directly from
the `errorSource <LearningProjection.errorSource>`, and receives a **target** stimulus from the process or system to
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

LearningProjections are executed after all of the mechanisms in a process or system have executed, including the
MonitoringMechanisms that provide the `error_signal` to each LearningProjection.  When the LearningProjection is
executed, it uses its `error_signal` to calculate changes to the `matrix <MappingProjection.MappingProjection.matrix>`
of its `mappingProjection`. Changes to the matrix are calculated so as to reduce the `error_signal`. The changes are
assigned as the `value <LearningProjection.value>` of the LearningProjection, but are not applied to
the `matrix <MappingProjection.MappingProjection.matrix>` until the next time the `mappingProjection` is executed
(see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import _objective_mechanism_role
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
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


WEIGHT_CHANGE_PARAMS = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

TARGET_ERROR = "TARGET_ERROR"
TARGET_ERROR_MEAN = "TARGET_ERROR_MEAN"
TARGET_ERROR_SUM = "TARGET_ERROR_SUM"
TARGET_SSE = "TARGET_SSE"
TARGET_MSE = "TARGET_MSE"


DefaultTrainingMechanism = ObjectiveMechanism

class LearningProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningProjection(Projection_Base):
    """
    LearningProjection(                      \
                 sender=None,                \
                 receiver=None,              \
                 function=BackPropagation    \
                 params=None,                \
                 name=None,                  \
                 prefs=None)


    Implements a projection that modifies the matrix param of a MappingProjection.

    COMMENT:
        Description:
            The LearningProjection class is a componentType in the Projection category of Function.
            It implements a projection to the parameterState of a MappingProjection that modifies its matrix parameter.
            It's execute method takes the output of a MonitoringMechanism (self.variable), and the input and output of
                the ProcessingMechanism to which its receiver MappingProjection projects, and generates a matrix of
                weight changes for the MappingProjection's matrix parameter.

        Class attributes:
            + className = LEARNING_PROJECTION
            + componentType = PROJECTION
            # + defaultSender (State)
            # + defaultReceiver (State)
            + paramClassDefaults (dict):
                + FUNCTION (Function): (default: BP)
                + FUNCTION_PARAMS:
                    + LEARNING_RATE (value): (default: 1)
            + paramNames (dict)
            + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            function (computes function specified in params[FUNCTION]
    COMMENT

    Arguments
    ---------
    sender : Optional[MonitoringMechanism or OutputState of one]
        the source of the `error_signal` for the LearningProjection. If it is not specified, one will be
        `automatically created <LearningProjection_Automatic_Creation>` that is appropriate for the
        LearningProjection's `errorSource <LearningProjection.errorSource>`.

    receiver : Optional[MappingProjection or ParameterState for ``matrix`` parameter of one]
        the `parameterState <ParameterState>` (or the `MappingProjection` that owns it) for the
        `matrix <MappingProjection.MappingProjection.matrix>` to be modified by the LearningProjection.

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

    sender : OutputState of MonitoringMechanism
        source of `error_signal`.

    receiver : ParameterState of MappingProjection
        parameterState for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of
        the `mappingProjection` to be modified by the LearningProjection.

    mappingProjection : MappingProjection
        the `MappingProjection` that owns the `parameterState <ParameterState>` to which the
        LearningProjection projects (i.e., that owns its `receiver <LearningProjection.receiver>`).

    mappingWeightMatrix : 2d np.array
        the `matrix <MappingProjection.MappingProjection.matrix>` parameter to be modified by learning
        (i.e., that belongs to the `mappingProjection`).

    errorSource : ProcessingMechanism
        the mechanism to which `mappingProjection` projects, and that is used to calculate the `error_signal`.

    error_signal : 1d np.array
        output of `errorSource <LearningProjection.errorSource>` (`sender <LearningProjection.sender>`) used as the
        input for the LearningProjection's `function <LearningProjection.function>`, to determine changes to the
        `mappingWeightMatrix`.

    variable : 1d np.array
        COMMENT:
            WRONG?  CORRECTED BELOW
            same as :py:data:`mappingProjection <LearningProjection.mappingProjection>`.
        COMMENT
        same as `error_signal`.

    weightChangeMatrix : 2d np.array
        matrix of changes to be made to the `mappingWeightMatrix` (rows correspond to sender, columns to receiver).

    value : 2d np.array
        same as `weightChangeMatrix`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for projection.
        Specified in the `prefs` argument of the constructor for the projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentType = LEARNING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # variableClassDefault = [[0],[0],[0]]x

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_SENDER: MonitoringMechanism_Base,
                               PARAMETER_STATES: None, # This suppresses parameterStates
                               WEIGHT_CHANGE_PARAMS:  # Determine how weight changes are applied to weight matrix
                                   {                  # Note:  assumes MappingProjection.function is LinearCombination
                                       FUNCTION_PARAMS: {OPERATION: SUM},
                                       PARAMETER_MODULATION_OPERATION: ModulationOperation.ADD,
                                       PROJECTION_TYPE: LEARNING_PROJECTION}
                               })

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=BackPropagation(learning_rate=1,
                                          activation_function=Logistic),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function, params=params)

        # Store args for deferred initialization
        self.init_args = locals().copy()
        self.init_args['context'] = self
        self.init_args['name'] = name

        # Flag for deferred initialization
        self.value = DEFERRED_INITIALIZATION

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure sender is a MonitoringMechanism or ProcessingMechanism and receiver is a ParameterState or
        MappingProjection

        Validate send in params[PROJECTION_SENDER] or, if not specified, sender arg:
        - must be the outputState of a MonitoringMechanism (e.g., ComparatorMechanism or WeightedErrorMechanism)
        - must be a list or 1D np.array (i.e., the format of an error_signal)

        Validate receiver in params[PARAMETER_STATES] or, if not specified, receiver arg:
        - must be either a MappingProjection or parameterStates[MATRIX]

         """

        # Parse params[PROJECTION_SENDER] if specified, and assign self.sender
        super()._validate_params(request_set, target_set, context)

        # VALIDATE SENDER
        sender = self.sender
        self._validate_sender(sender)

        # VALIDATE RECEIVER
        try:
            receiver = target_set[PARAMETER_STATES]
            self._validate_receiver(receiver)
        except (KeyError, LearningProjectionError):
            # PARAMETER_STATES not specified:
            receiver = self.receiver
            self._validate_receiver(receiver)

        # VALIDATE WEIGHT CHANGE PARAMS
        try:
            weight_change_params = target_set[WEIGHT_CHANGE_PARAMS]
        except KeyError:
            pass
        else:
            # FIX: CHECK THAT EACH ONE INCLUDED IS A PARAM OF A LINEAR COMBINATION FUNCTION
            for param_name, param_value in weight_change_params.items():
                if param_name is FUNCTION:
                    raise LearningProjectionError("{} of {} contains a function specification ({}) that would override"
                                              " the LinearCombination function of the targeted MappingProjection".
                                              format(WEIGHT_CHANGE_PARAMS,
                                                     self.name,
                                                     param_value))

    def _validate_sender(self, sender):
        """Make sure sender is a MonitoringMechanism or ProcessingMechanism or the outputState for one;
        """

        # If specified as a MonitoringMechanism, reassign to its outputState
        if isinstance(sender, MonitoringMechanism_Base):
            sender = self.sender = sender.outputState

        # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
        if isinstance(sender, OutputState):
            if not isinstance(sender.value, (list, np.ndarray)):
                raise LearningProjectionError("Sender for {} (outputState of MonitoringMechanism {}) "
                                          "must be a list or 1D np.array".format(self.name, sender))
            if not np.array(sender.value).ndim == 1:
                raise LearningProjectionError("OutputState of MonitoringMechanism ({}) for {} must be an 1D np.array".
                                          format(sender, self.name))

        # If specification is a MonitoringMechanism class, pass (it will be instantiated in _instantiate_sender)
        elif inspect.isclass(sender) and issubclass(sender,  MonitoringMechanism_Base):
            pass

        else:
            raise LearningProjectionError("Sender arg (or {} param ({}) for must be a MonitoringMechanism, "
                                      "the outputState of one, or a reference to the class"
                                      .format(PROJECTION_SENDER, sender, self.name, ))

    def _validate_receiver(self, receiver):
        """Make sure receiver is a MappingProjection or the parameterState of one
        """

        if not isinstance(receiver, (MappingProjection, ParameterState)):
            raise LearningProjectionError("Receiver arg ({}) for {} must be a MappingProjection or a parameterState of one"
                                      .format(receiver, self.name))
        # If it is a parameterState and the receiver already has a parameterStates dict
        #     make sure the assignment is to its MATRIX entry
        if isinstance(receiver, ParameterState):
            if receiver.owner.parameterStates and not receiver is receiver.owner.parameterStates[MATRIX]:
                raise LearningProjectionError("Receiver arg ({}) for {} must be the {} parameterState of a"
                                          "MappingProjection".format(receiver, self.name, MATRIX, ))

        # Notes:
        # * if specified as a MappingProjection, it will be assigned to a parameter state in _instantiate_receiver
        # * the value of receiver will be validated in _instantiate_receiver

    def _instantiate_attributes_before_function(self, context=None):
        """Override super to call _instantiate_receiver before calling _instantiate_sender

        Call _instantiate_receiver first since both _instantiate_sender and _instantiate_function
            reference the MappingProjection's weight matrix: self.mappingProjection.matrix

        """
        # FIX: PROBLEM: _instantiate_receiver usually follows _instantiate_function,
        # FIX:          and uses self.value (output of function) to validate against receiver.variable

        self._instantiate_receiver(context)

        # # MODIFIED 8/14/16: COMMENTED OUT SINCE SOLVED BY MOVING add_to TO _instantiate_attributes_after_function
        # # "Cast" self.value to MappingProjection parameterState's variable to pass validation in _instantiate_sender
        # # Note: this is because _instantiate_sender calls _add_projection_to
        # # (since self.value is not assigned until _instantiate_function; it will be reassigned there)
        # self.value = self.receiver.variable

        super()._instantiate_attributes_before_function(context)

    def _instantiate_attributes_after_function(self, context=None):
        """Override super since it calls _instantiate_receiver which has already been called above
        """
        # pass
        # MODIFIED 8/14/16: MOVED FROM _instantiate_sender
        # Add LearningProjection to MappingProjection's parameterState
        # Note: needs to be done after _instantiate_function, since validation requires self.value be assigned
        self.add_to(receiver=self.mappingProjection, state=self.receiver, context=context)

    def _instantiate_receiver(self, context=None):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a MappingProjection, assign LearningProjection to parameterStates[MATRIX]
            for the projection;  if that does not exist, instantiate and assign as the receiver for the LearningProjection
        If specified as a ParameterState, validate that it is parameterStates[MATRIX]
        Validate that the LearningProjection's error matrix is the same shape as the recevier's weight matrix
        Re-assign LearningProjection's variable to match the height (number of rows) of the matrix

        Notes:
        * This must be called before _instantiate_sender since that requires access to self.receiver
            to determine whether to use a ComparatorMechanism or <MappingProjection>.receiverError for error signals
        * Doesn't call super()._instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls _add_projection_to_mechanism
        """

# FIX: ??REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to
# FIX: OR, MAKE SURE FUNCTIONALITY IS COMPARABLE

        weight_change_params = self.paramsCurrent[WEIGHT_CHANGE_PARAMS]

        # VALIDATE that self.receiver is a ParameterState or a MappingProjection

        # If receiver is a ParameterState, and receiver's parameterStates dict has been instantiated,
        #    make sure LearningProjection is being assigned to the parameterStates[MATRIX] of a MappingProjection
        if isinstance(self.receiver, ParameterState):

            self.mappingProjection = self.receiver.owner

            # MODIFIED 10/29/16 OLD:
            # Reciever must be a MappingProjection with a LinearCombination function
            if not isinstance(self.mappingProjection, MappingProjection):
                raise LearningProjectionError("Receiver arg ({}) for {} must be the parameterStates[{}] "
                                          "of a MappingProjection (rather than a {})".
                                          format(self.receiver,
                                                 self.name,
                                                 MATRIX,
                                                 self.mappingProjection.__class__.__name__))
            if not isinstance(self.receiver.function.__self__, LinearCombination):
                raise LearningProjectionError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
                                          format(self.receiver,
                                                 self.name,
                                                 LINEAR_COMBINATION_FUNCTION,
                                                 self.mappingProjection.function.__self__.__class__.__name__))

            # # MODIFIED 10/29/16 NEW:
            # # Reciever must be the parameterState for a MappingProjection with a LinearCombination identity function
            # if not isinstance(self.mappingProjection, MappingProjection):
            #     raise LearningProjectionError("Receiver arg ({}) for {} must be the parameterStates[{}] "
            #                               "of a MappingProjection (rather than a {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      MATRIX,
            #                                      self.mappingProjection.__class__.__name__))
            # if not isinstance(self.receiver.function.__self__, LinearCombination):
            #     raise LearningProjectionError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      LINEAR_FUNCTION,
            #                                      self.mappingProjection.function.__self__.__class__.__name__))
            # # MODIFIED 10/29/16 END


            # receiver is parameterState[MATRIX], so update its params with ones specified by LearningProjection
            # (by default, change LinearCombination.operation to SUM paramModulationOperation to ADD)
            if (self.mappingProjection.parameterStates and
                    self.receiver is self.mappingProjection.parameterStates[MATRIX]):
                self.receiver.paramsCurrent.update(weight_change_params)

            else:
                raise LearningProjectionError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] param of the receiver".
                                          format(self.receiver, self.name, MATRIX))

        # Receiver was specified as a MappingProjection
        elif isinstance(self.receiver, MappingProjection):

            self.mappingProjection = self.receiver

            from PsyNeuLink.Components.States.InputState import _instantiate_state_list
            from PsyNeuLink.Components.States.InputState import _instantiate_state

            # Check if MappingProjection has parameterStates Ordered Dict and MATRIX entry
            try:
                self.receiver.parameterStates[MATRIX]

            # receiver does NOT have parameterStates attrib
            except AttributeError:
                # Instantiate parameterStates Ordered dict
                #     with ParameterState for receiver's functionParams[MATRIX] param
                self.receiver.parameterStates = _instantiate_state_list(owner=self.receiver,
                                                                       state_list=[(MATRIX,
                                                                                    weight_change_params)],
                                                                       state_type=ParameterState,
                                                                       state_param_identifier=PARAMETER_STATE,
                                                                       constraint_value=self.mappingWeightMatrix,
                                                                       constraint_value_name=LEARNING_PROJECTION,
                                                                       context=context)
                self.receiver = self.receiver.parameterStates[MATRIX]

            # receiver has parameterStates but not (yet!) one for MATRIX, so instantiate it
            except KeyError:
                # Instantiate ParameterState for MATRIX
                self.receiver.parameterStates[MATRIX] = _instantiate_state(owner=self.receiver,
                                                                            state_type=ParameterState,
                                                                            state_name=MATRIX,
                                                                            state_spec=PARAMETER_STATE,
                                                                            state_params=weight_change_params,
                                                                            constraint_value=self.mappingWeightMatrix,
                                                                            constraint_value_name=LEARNING_PROJECTION,
                                                                            context=context)

            # receiver has parameterState for MATRIX, so update its params with ones specified by LearningProjection
            else:
                # MODIFIED 8/13/16:
                # FIX: ?? SHOULD THIS USE _assign_defaults:
                self.receiver.parameterStates[MATRIX].paramsCurrent.update(weight_change_params)

            # Assign self.receiver to parameterState used for weight matrix param
            self.receiver = self.receiver.parameterStates[MATRIX]

        # If it is not a ParameterState or MappingProjection, raise exception
        else:
            raise LearningProjectionError("Receiver arg ({}) for {} must be a MappingProjection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        if DEFERRED_DEFAULT_NAME in self.name:
            self.name = self.mappingProjection.name + ' ' + self.componentName
            # self.name = self.mappingProjection.name + \
            #             self.mappingProjection.parameterStates[MATRIX].name + \
            #             ' ' + self.componentName

        # Assign errorSource as the MappingProjection's receiver mechanism
        self.errorSource = self.mappingProjection.receiver.owner

        # GET RECEIVER'S WEIGHT MATRIX
        self._get_mapping_projection_weight_matrix()

        # Format input to MappingProjection's weight matrix
        # MODIFIED 8/19/16:
        # self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])
        self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])

        # Format output of MappingProjection's weight matrix
        # Note: this is used as a template for output value of its receiver mechanism (i.e., to which it projects)
        #       but that may not yet have been instantiated;  assumes that format of input = output for receiver mech
        # MODIFIED 8/19/16:
        # self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])
        self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])

    def _get_mapping_projection_weight_matrix(self):
        """Get weight matrix for MappingProjection to which LearningProjection projects

        """

        message = "PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".format(self.receiver.name,
                                                                                             FUNCTION_PARAMS,
                                                                                             MATRIX)
        if isinstance(self.receiver, ParameterState):
            try:
                self.mappingWeightMatrix = self.mappingProjection.matrix
            except KeyError:
                raise LearningProjection(message)

        elif isinstance(self.receiver, MappingProjection):
            try:
                self.mappingWeightMatrix = self.receiver.matrix
            except KeyError:
                raise LearningProjection(message)

    def _instantiate_sender(self, context=None):
        # DOCUMENT: SEE UPDATE BELOW
        """Assign self.variable to MonitoringMechanism output or self.receiver.receivererror_signals

        Call this after _instantiate_receiver, as that is needed to determine the sender (i.e., source of error_signal)

        If sender arg or PROJECTION_SENDER was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
            - assign its outputState.value as self.variable
        If sender was not specified (i.e., passed as MonitoringMechanism_Base specified in paramClassDefaults):
           if the owner of the MappingProjection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
               - assign its outputState.value as self.variable
           UPDATE: otherwise, if MappingProjection's receiver has an error_signal attribute, use that as self.variable
               (e.g., "hidden units in a multilayered neural network, using BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception

FROM TODO:
#    - _instantiate_sender:
#        - examine mechanism to which MappingProjection projects:  self.receiver.owner.receiver.owner
#            - check if it is a terminal mechanism in the system:
#                - if so, assign:
#                    - ComparatorMechanism MonitoringMechanism
#                        - ProcessInputState for ComparatorMechanism (name it??) with projection to target inputState
#                        - MappingProjection from terminal ProcessingMechanism to LinearCompator sample inputState
#                - if not, assign:
#                    - WeightedSum MonitoringMechanism
#                        - MappingProjection from preceding MonitoringMechanism:
#                            preceding processing mechanism (ppm):
#                                ppm = self.receiver.owner.receiver.owner
#                            preceding processing mechanism's output projection (pop)
#                                pop = ppm.outputState.projections[0]
#                            preceding processing mechanism's output projection learning signal (popls):
#                                popls = pop.parameterState.receivesFromProjections[0]
#                            preceding MonitoringMechanism (pem):
#                                pem = popls.sender.owner
#                            assign MappingProjection from pem.outputState to self.inputState
#                        - Get weight matrix for pop (pwm):
#                                pwm = pop.parameterState.params[MATRIX]

# HAS SENDER:
    # VALIDATE
# HAS NO SENDER:
    # self.errorSource PROJECTS TO A MONITORING MECHANISM
    #         assign it as sender
    # self.errorSource DOESN'T PROJECT TO A MONITORING MECHANISM
        # self.errorSource PROJECTS TO A PROCESSING MECHANISM:
            # INSTANTIATE WeightedSum MonitoringMechanism
        # self.errorSource PROJECTS DOESN'T PROJECT TO A PROCESSING MECHANISM:
            # INSTANTIATE DefaultTrainingMechanism

        """
# IMPLEMENT: rename .monitoringMechanism -> .objective_mechanism
#            rename .errorSource -> .error_source

        # FIX: 8/7/16
        # FIX: NEED TO DEAL HERE WITH CLASS SPECIFICATION OF MonitoringMechanism AS DEFAULT
        # FIX: OR HAVE ALREADY INSTANTIATED DEFAULT MONITORING MECHANISM BEFORE REACHING HERE
        # FIX: EMULATE HANDLING OF DefaultMechanism (for MappingProjection) AND DefaultController (for ControlProjection)

        # FIX: 8/18/16
        # FIX: ****************
        # FIX: ASSIGN monitoring_source IN ifS, NOT JUST else
        # FIX: SAME FOR self.errorSource??

        objective_mechanism = None

        # ObjectiveMechanism specified for sender, so re-assign to its outputState
        if isinstance(self.sender, ObjectiveMechanism):
            self.sender = self.sender.outputState

        # OutputState specified (or re-assigned) for sender
        if isinstance(self.sender, OutputState):
            # Validate that it belongs to an ObjectiveMechanism being used for learning
            if not _objective_mechanism_role(self.sender.owner, LEARNING):
                raise LearningProjectionError("OutputState ({}) specified as sender for {} belongs to a {}"
                                          " rather than an ObjectiveMechanism with role=LEARNING".
                                          format(self.sender.name,
                                                 self.name,
                                                 self.sender.owner.__class__.__name__))
            self._validate_error_signal(self.sender.value)

            # - assign ObjectiveMechanism's outputState.value as self.variable
            # FIX: THIS DOESN"T SEEM TO HAPPEN HERE.  DOES IT HAPPEN LATER??

            # Add reference to ObjectiveMechanism to MappingProjection
            objective_mechanism = self.sender

        # ObjectiveMechanism class (i.e., not an instantiated object) specified for sender, so instantiate it:
        # - for terminal mechanism of Process, instantiate with Comparator function
        # - for preceding mechanisms, instantiate with WeightedError function
        else:
            # Get errorSource:  ProcessingMechanism for which error is being monitored
            #    (i.e., the mechanism to which the MappingProjection projects)
            # Note: MappingProjection._instantiate_receiver has not yet been called, so need to do parse below
            from PsyNeuLink.Components.States.InputState import InputState
            if isinstance(self.mappingProjection.receiver, Mechanism):
                self.errorSource = self.mappingProjection.receiver
            elif isinstance(self.mappingProjection.receiver, InputState):
                self.errorSource = self.mappingProjection.receiver.owner

            next_level_objective_mech_output = None

            # Check if errorSource has a projection to an ObjectiveMechanism or some other type of ProcessingMechanism
            for projection in self.errorSource.outputState.sendsToProjections:
                # errorSource has a projection to an ObjectiveMechanism being used for learning,
                #  so validate it, assign it, and quit search
                if _objective_mechanism_role(projection.receiver.owner, LEARNING):
                    self._validate_error_signal(projection.receiver.owner.outputState.value)
                    objective_mechanism = projection.receiver.owner
                    break
                # errorSource has a projection to a ProcessingMechanism, so:
                #   - determine whether that has a LearningProjection
                #   - if so, get its MonitoringMechanism and weight matrix (needed by BackProp)
                if isinstance(projection.receiver.owner, ProcessingMechanism_Base):
                    try:
                        next_level_learning_projection = projection.parameterStates[MATRIX].receivesFromProjections[0]
                    except (AttributeError, KeyError):
                        # Next level's projection has no parameterStates, Matrix parameterState or projections to it
                        #    => no LearningProjection
                        pass # FIX: xxx ?? ADD LearningProjection here if requested?? or intercept error message to do so?
                    else:
                        # Next level's projection has a LearningProjection so get:
                        #     the weight matrix for the next level's projection
                        #     the MonitoringMechanism that provides error_signal
                        # next_level_weight_matrix = projection.matrix
                        next_level_objective_mech_output = next_level_learning_projection.sender

            # errorSource does not project to an ObjectiveMechanism used for learning
            if not objective_mechanism:

                # FIX:  NEED TO DEAL WITH THIS RE: RL -> DON'T CREATE BACK PROJECTIONS??
                # NON-TERMINAL Mechanism
                # errorSource at next level projects to a MonitoringMechanism:
                #    instantiate ObjectiveMechanism configured with WeightedError Function
                #    (computes contribution of each element in errorSource to error at level to which it projects)
                #    and the back-projection for its error signal:
                if next_level_objective_mech_output:
                    error_signal = np.zeros_like(next_level_objective_mech_output.value)
                    next_level_output = projection.receiver.owner.outputState
                    activity = np.zeros_like(next_level_output.value)
                    matrix=projection.parameterStates[MATRIX]
                    derivative = next_level_objective_mech_output.sendsToProjections[0].\
                        receiver.owner.receiver.owner.function_object.derivative
                    from PsyNeuLink.Components.Functions.Function import WeightedError
                    objective_mechanism = ObjectiveMechanism(monitored_values=[next_level_output,
                                                                       next_level_objective_mech_output],
                                                              names=['ACTIVITY','ERROR_SIGNAL'],
                                                              function=ErrorDerivative(variable_default=[activity,
                                                                                                       error_signal],
                                                                                       derivative=derivative),
                                                              role=LEARNING,
                                                              name=self.mappingProjection.name + " Error_Derivative")
                # TERMINAL Mechanism
                # errorSource at next level does NOT project to an ObjectiveMechanism:
                #     instantiate ObjectiveMechanism configured as a comparator
                #         that compares errorSource output with external training signal
                else:
                    # Instantiate ObjectiveMechanism to receive the (externally provided) target for training
                    try:
                        sample_state_name = self.errorSource.paramsCurrent[MONITOR_FOR_LEARNING]
                        sample_source = self.errorSource.outputStates[sample_state_name]
                        sample_size = np.zeros_like(sample_source)
                    except KeyError:
                        # No state specified so use Mechanism as sender arg
                        sample_source = self.errorSource
                        sample_size = np.zeros_like(self.errorSource.outputState.value)

                    # Assign output_signal to output of errorSource
                    if self.function.componentName is BACKPROPAGATION_FUNCTION:
                        target_size = np.zeros_like(self.errorSource.outputState.value)
                    # Force sample and target of Comparartor to be scalars for RL
                    elif self.function.componentName is RL_FUNCTION:
                        sample_size = np.array([0])
                        target_size = np.array([0])
                    else:
                        raise LearningProjectionError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                                  format(self.function.name, self.name))

                    # IMPLEMENTATION NOTE: specify target as a template value (matching the sample's output value)
                    #                      since its projection (from either a ProcessInputState or a SystemInputState)
                    #                      will be instantiated by the Composition object to which the mechanism belongs
                    # FIX: FOR RL, NEED TO BE ABLE TO CONFIGURE OBJECTIVE MECHANISM WITH SCALAR INPUTSTATES
                    # FIX:         AND FULL CONNECTIVITY MATRICES FROM THE MONITORED OUTPUTSTATES
                    objective_mechanism = ObjectiveMechanism(default_input_value=[sample_size, target_size],
                                                             monitored_values=[sample_source, target_size],
                                                             names=[SAMPLE,TARGET],
                                                             # FIX: WHY DO WEIGHTS HAVE TO BE AN ARRAY HERE??
                                                             function=LinearCombination(weights=[[-1], [1]]),
                                                             role=LEARNING,
                                                             params= {OUTPUT_STATES:
                                                                          [{NAME:TARGET_ERROR},
                                                                           {NAME:TARGET_ERROR_MEAN,
                                                                            CALCULATE:lambda x: np.mean(x)},
                                                                           {NAME:TARGET_ERROR_SUM,
                                                                            CALCULATE:lambda x: np.sum(x)},
                                                                           {NAME:TARGET_SSE,
                                                                            CALCULATE:lambda x: np.sum(x*x)},
                                                                           {NAME:TARGET_MSE,
                                                                            CALCULATE:lambda x: np.sum(x*x)/len(x)}]},
                                                             name=self.mappingProjection.name + " Target_Error")
                    objective_mechanism.learning_role = TARGET

            self.sender = objective_mechanism.outputState

            # "Cast" self.variable to match value of sender (MonitoringMechanism) to pass validation in add_to()
            # Note: self.variable will be re-assigned in _instantiate_function()
            self.variable = self.error_signal

            # Add self as outgoing projection from MonitoringMechanism
            from PsyNeuLink.Components.Projections.Projection import _add_projection_from
            _add_projection_from(sender=objective_mechanism,
                                state=objective_mechanism.outputState,
                                projection_spec=self,
                                receiver=self.receiver,
                                context=context)

        # VALIDATE THAT OUTPUT OF SENDER IS SAME LENGTH AS THIRD ITEM (ERROR SIGNAL) OF SEL.FFUNCTION.VARIABLE

        # Add reference to MonitoringMechanism to MappingProjection
        self.mappingProjection.monitoringMechanism = objective_mechanism

    def _validate_error_signal(self, error_signal):
        """Check that error signal (MonitoringMechanism.outputState.value) conforms to what is needed by self.function
        """

        if self.function.componentName is RL_FUNCTION:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must == 1
            #     (since error signal is a scalar for RL)
            if len(error_signal) != 1:
                raise LearningProjectionError("Length of error signal ({}) received by {} from {}"
                                          " must be 1 since {} uses {} as its learning function".
                                          format(len(error_signal), self.name, self.sender.owner.name, self.name, RL_FUNCTION))
        if self.function.componentName is BACKPROPAGATION_FUNCTION:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must be the
            #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)
            if len(error_signal) != self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]:
                raise LearningProjectionError("Length of error signal ({}) received by {} from {} must match the"
                                          "receiver dimension ({}) of the weight matrix for {}".
                                          format(len(error_signal),
                                                 self.name,
                                                 self.sender.owner.name,
                                                 len(self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]),
                                                 self.mappingProjection))
        else:
            raise LearningProjectionError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                      format(self.function.name, self.name))

    def _instantiate_function(self, context=None):
        """Construct self.variable for input to function, call super to instantiate it, and validate output

        function implements function to compute weight change matrix for receiver (MappingProjection) from:
        - input: array of sender values (rows) to MappingProjection weight matrix (self.variable[0])
        - output: array of receiver values (cols) for MappingProjection weight matrix (self.variable[1])
        - error:  array of error signals for receiver values (self.variable[2])
        """

        # Reconstruct self.variable as input for function
        self.variable = [[0]] * 3
        self.variable[0] = self.input_to_weight_matrix
        self.variable[1] = self.output_of_weight_matrix
        self.variable[2] = self.error_signal

        super()._instantiate_function(context)

        from PsyNeuLink.Components.Functions.Function import ACTIVATION_FUNCTION, TransferFunction
        # Insure that the learning function is compatible with the activation function of the errorSource
        error_source_activation_function_type = type(self.errorSource.function_object)
        function_spec = self.function_object.paramsCurrent[ACTIVATION_FUNCTION]
        if isinstance(function_spec, TransferFunction):
            learning_function_activation_function_type = type(function_spec)
        elif issubclass(function_spec, TransferFunction):
            learning_function_activation_function_type = function_spec
        else:
            raise LearningProjectionError("PROGRAM ERROR: activation function ({}) for {} is not a TransferFunction".
                                      format(function_spec, self.name))
        if error_source_activation_function_type != learning_function_activation_function_type:
            raise LearningProjectionError("Activation function ({}) of error source ({}) is not compatible with "
                                      "the activation function ({}) specified for {}'s function ({}) ".
                                      format(error_source_activation_function_type.__name__,
                                             self.errorSource.name,
                                             learning_function_activation_function_type.__name__,
                                             self.name,
                                             self.params[FUNCTION].__self__.__class__.__name__))

        # FIX: MOVE TO AFTER INSTANTIATE FUNCTION??
        # IMPLEMENTATION NOTE:  MOVED FROM _instantiate_receiver
        # Insure that LearningProjection output (error signal) and receiver's weight matrix are same shape
        try:
            receiver_weight_matrix_shape = self.mappingWeightMatrix.shape
        except TypeError:
            # self.mappingWeightMatrix = 1
            receiver_weight_matrix_shape = 1
        try:
            LEARNING_PROJECTION_shape = self.value.shape
        except TypeError:
            LEARNING_PROJECTION_shape = 1

        if receiver_weight_matrix_shape != LEARNING_PROJECTION_shape:
            raise ProjectionError("Shape ({0}) of matrix for {1} learning signal from {2}"
                                  " must match shape of receiver weight matrix ({3}) for {4}".
                                  format(LEARNING_PROJECTION_shape,
                                         self.name,
                                         self.sender.name,
                                         receiver_weight_matrix_shape,
                                         # self.receiver.owner.name))
                                         self.mappingProjection.name))

    def execute(self, input=None, clock=CentralClock, time_scale=None, params=None, context=None):
    # def execute(self, input=None, params=None, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):
        """
        DOCUMENT:
        LearnningSignal (Projection):
            - sender:  output of Monitoring Mechanism
                default: receiver.owner.outputState.sendsToProjections.<MonitoringMechanism> if specified,
                         else default ComparatorMechanism
            - receiver: MappingProjection parameterState (or some equivalent thereof)

        MappingProjection should have LEARNING_PARAM which:
           - specifies LearningProjection
           - defaults to BP
        ?? - uses self.outputStates.sendsToProjections.<MonitoringMechanism> if specified


        :return: (2D np.array) self.weightChangeMatrix
        """

        # Pass during initialization (since has not yet been fully initialized
        if self.value is DEFERRED_INITIALIZATION:
            return self.value

        # GET INPUT TO Projection to Error Source:
        # Array of input values from MappingProjection's sender mechanism's outputState
        input = self.mappingProjection.sender.value

        # ASSIGN OUTPUT TO ERROR SOURCE
        # Array of output values for MappingProjection's receiver mechanism
        # output = self.mappingProjection.receiver.owner.outputState.value
# FIX: IMPLEMENT self.unconvertedOutput AND self.convertedOutput, VALIDATE QUANTITY BELOW IN _instantiate_sender,
# FIX:   ASSIGN self.input ACCORDINGLY
        output = self.errorSource.outputState.value

        # ASSIGN ERROR
# FIX: IMPLEMENT self.input AND self.convertedInput, VALIDATE QUANTITY BELOW IN _instantiate_sender, ASSIGN ACCORDINGLY
        error_signal = self.error_signal

        # CALL function TO GET WEIGHT CHANGES
        # rows:  sender errors;  columns:  receiver errors
        self.weightChangeMatrix = self.function([input, output, error_signal], params=params, context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.weightChangeMatrix))

        self.value = self.weightChangeMatrix

        # # TEST PRINT
        # print("\nr### WEIGHT CHANGES FOR {} TRIAL {}:\n{}".format(self.name, CentralClock.trial, self.value))

        return self.value

    @property
    def error_signal(self):
        return self.sender.value

    @property
    def monitoringMechanism(self):
        return self.sender.owner
