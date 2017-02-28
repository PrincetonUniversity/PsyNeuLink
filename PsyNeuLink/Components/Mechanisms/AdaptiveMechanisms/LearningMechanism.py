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

LearningMechanism is a subtype of the AdaptiveMechanism Type of the Mechanism Category of Component
It implements a mechanism that calculates changes to a projection's parameters.
It's function takes the output of an ObjectiveMechanism (self.variable) and generates a
learning_signal (2d arry of parameter changes) to be used by the recipient of a LearningProjection
that projects from the LearningMechanism to a MappingProjection.


COMMENT:
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

COMMENT

.. _LearningMechanism_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import ComparatorMechanism, \
                                                                                      COMPARATOR_SAMPLE
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.WeightedErrorMechanism import WeightedErrorMechanism, \
                                                                                         PROJECTION_TO_NEXT_MECHANISM
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
ERROR_SIGNAL_INDEX = 2

# Used to name inputStates:
ACTIVATION_INPUT = 'activation_input'
ACTIVATION_OUTPUT = 'activation_output'
ERROR_SIGNAL = 'error_signal'
input_state_names = [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]

# Argument names:
ERROR_MATRIX = 'error_matrix'

# Name of outputState:
LEARNING_SIGNAL = 'learning_signal'

WEIGHT_CHANGE_PARAMS = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

TARGET_ERROR = "TARGET_ERROR"
TARGET_ERROR_MEAN = "TARGET_ERROR_MEAN"
TARGET_ERROR_SUM = "TARGET_ERROR_SUM"
TARGET_SSE = "TARGET_SSE"
TARGET_MSE = "TARGET_MSE"


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
                 error_matrix                \
                 function=BackPropagation    \
                 learning_rate=None          \
                 params=None,                \
                 name=None,                  \
                 prefs=None)

    Implements a projection that modifies the matrix param of a MappingProjection.

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
        take three items specifying: 1) the input to the `MappingProjection` being learned, 2) the resulting output of
        the mechanism to which it projects, and 3) the error signal associated with that mechanism's output (see
        `variable <LearningMechanism.variable>` for details).

    COMMENT
        activation_derivative : Function or function
            specifies the derivative of the function of the mechanism that receives the `MappingProjection` being learned
            (see `activation_derivative` for details).
    COMMENT

    error_matrix : List, 2d np.array, ParameterState or MappingProjection
        specifies the matrix used to generate the `error_signal` (see `error_matrix` for details).

    function : LearningFunction or function
        specifies the function used to compute the `learning_signal` (see `function <LearningMechanism.function>` for
        details).

    learning_rate : float
        specifies the learning_rate for this LearningMechanism (see `learning_rate <LearningMechanism.learning_rate>`
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
        has three items, each of which is a 1d np.array: `activation_input`, `activation_output`, and `error_signal`.

    activation_input : 1d np.array
        the input to the `MappingProjection` being learned.

    activation_output : 1d np.array
        the output of the mechanism that receives the `MappingProjection` being learned.

    COMMENT:
        activation_derivative : Function or function
            the derivative of the function of the mechanism that receives the `MappingProjection` being learned.
    COMMENT

    error_signal : 1d np.array
        the error signal, typically generated by an ObjectiveMechanism used to evalue the output of the next mechanism
        in the the process or system (i.e., the one projected to by the mechanism that receives the MappingProjection
        being learned).

    error_matrix : List or 2d np.array
        the matrix for the `MappingProjection` that projects *from* the mechanism that receives the MappingProjection
        being learned, *to* the next mechanism in the process or system, from which the `error_signal` was generated.

    function : LearningFunction or function : default BackPropagation
        specifies function used to compute the `learning_signal`.  Must take the following arguments:
        `input` (list or 1d array), `output` (list or 1d array), `derivative` (function) and `error` (list or 1d array).

    learning_rate : float : default 1.0
        specifies the learning_rate for this LearningMechanism; it is superceded by the learning_rate for the
        process or system if either of those is specified (see
        ` process learning_rate <Process.Process_Base.learning_rate>` and
        ` system learning_rate <System.System_Base.learning_rate>` for details).

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
        OUTPUT_STATES:[{NAME:LEARNING_SIGNAL}]
    })

    @tc.typecheck
    def __init__(self,
                 variable:tc.any(list, np.ndarray),
                 error_matrix:tc.optional(tc.any(list, np.ndarray, ParameterState, MappingProjection))=None,
                 function:is_function_type=BackPropagation,
                 learning_rate:float=1.0,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(error_matrix=error_matrix,
                                                  function=function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name
        # delete self.init_args[ERROR_MATRIX]
        # delete self.init_args[LEARNING_RATE]

        # # Flag for deferred initialization
        # self.value = DEFERRED_INITIALIZATION

        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):

        super()._validate_variable(variable, context)

        # Validate that variable has exactly three items:  activation_input, activation_output, and error_signal
        if len(self.variable) != 3:
            raise LearningMechanismError("Variable for {} ({}) must have three items ({}, {}, and {})".
                                format(self.name, self.variable, ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL))

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
        """Validate error_matrix specification
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # IMPLEMENTATION NOTE:  REDUNDANT WITH typecheck?
        try:
            error_matrix = target_set[ERROR_MATRIX]
        except KeyError:
            raise LearningMechanismError("PROGRAM ERROR:  No specification for {} in {}".
                                format(ERROR_MATRIX, self.name))

        if not isinstance(error_matrix, (list, np.ndarray, ParameterState, MappingProjection)):
            raise LearningMechanismError("The {} arg for {} must be a list, 2d np.array, ParamaterState or "
                                          "MappingProjection".format(ERROR_MATRIX, self.name))

        if isinstance(error_matrix, MappingProjection):
            try:
                error_matrix = error_matrix.parameterStates[MATRIX]
            except KeyError:
                raise LearningMechanismError("The MappingProjection specified for the {} arg of {} ({})"
                                              "must have a {} paramaterState".
                                              format(ERROR_MATRIX, self.name, error_matrix, MATRIX))

        if isinstance(error_matrix, ParameterState):
            if np.array(error_matrix.value).ndim != 2:
                raise LearningMechanismError("The value of the {} parameterState specified for the {} arg of {} ({}) "
                                              "is not a 2d array (matrix)".
                                              format(MATRIX, ERROR_MATRIX, self.name, error_matrix))

    def _instantiate_attributes_before_function(self, context=None):
        """Parse error_matrix specification and insure it is compatible with error_signal and actiation_sample
        """
        super()._instantiate_attributes_before_function(context=context)


        activity_len = len(self.activation_output)
        error_len = len(self.error_signal)

        # Validate that activation_output and error_signal are the same length
        if activity_len != error_len:
            raise LearningMechanismError("Items {} ({}: {}) and {} ({}: {}) of variable for {} "
                                          "must be the same length".
                                          format(ACTIVATION_OUTPUT_INDEX,
                                                 ACTIVATION_OUTPUT,
                                                 self.variable[ACTIVATION_OUTPUT_INDEX],
                                                 ERROR_SIGNAL_INDEX,
                                                 ERROR_SIGNAL,
                                                 self.variable[ERROR_SIGNAL_INDEX],
                                                 self.name))

        if isinstance(self.error_matrix, MappingProjection):
            self.error_matrix = self.error_matrix.parameterStates[MATRIX]

        if isinstance(self.error_matrix, ParameterState):
            self.error_matrix = np.array(self.error_matrix.value)

        if self.error_matrix.ndim != 2:
            raise LearningMechanismError("\'matrix\' arg for {} must be 2d (it is {})".
                               format(self.__class__.__name__, matrix.ndim))

        cols = self.error_matrix.shape[1]
        if  cols != error_len:
            raise FunctionError("Number of columns ({}) of \'{}\' arg for {}"
                                     " must equal length of {} ({})".
                                     # format(cols,MATRIX, self.__class__.__name__,error_len))
                                     format(cols,MATRIX, self.name, ERROR_SIGNAL, error_len))

    def _instantiate_input_states(self, context=None):
        """Insure that inputState values are compatible with derivative functions and error_matrix
        """
        super()._instantiate_input_states(context=context)

    def _instantiate_attributes_after_function(self, context=None):

        super()._instantiate_attributes_after_function(context=context)

        # NEED TO CHECK COMPATIBILITY FOR THE FOLLOWING:
        #     ACTIVATION_INPUT ≌ 1st item of function variable
        #     ACTIVATION_OUTPUT ≌ 2nd item of function variable
        #     ERROR_INPUT ≌ 3rd item of function variable (error_signal (= np.dot(error.matrix, error_signal))


        #     error_matrix rows:  sender errors;  columns:  receiver errors
        #     BackPropagation learning algorithm (Generalized Delta Rule - :ref:`<LINK>`):
        #         weight = weight + (learningRate * error_derivative * activation_derivative * activation_output)
        #         for sumSquared error function:  error_derivative = (target - sample)
        #         for logistic activation function: activation_derivative = activation_output * (1-activation_output)
        #     NEEDS:
        #     - error_signal (from ObjectiveMechanism or equivalent)
        #     - errorDerivative:  get from error_source [??get from FUNCTION of ComparatorMechanism??]
        #     - transferDerivative:  get from function of error_source [??get from FUNCTION of Processing Mechanism]
        # FIX: Call _validate_error_signal HERE?? (GET FROM LearningProjection)

    def _instantiate_output_states(self, context=None):
        super()._instantiate_output_states(context=context)

        # OVERRIDE TO INSTANTIATE OUTPUTSTATE THAT TAKES FULL OUTPUT OF FUNCTION (MATRIX) AS ITS VALUE
        #     function output ≌ error_matrix

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute LearningMechanism function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # # Pass during initialization (since has not yet been fully initialized
        # if self.value is DEFERRED_INITIALIZATION:
        #     return self.value

        # COMPUTE WEIGHTED ERROR SIGNAL (weighted version of dE/dA):
        weighted_error_signal = np.dot(self.error_matrix, self.error_signal)

        # COMPUTE LEARNING SIGNAL (dE/dW):
        self.learning_signal = self.function(variable=[self.activation_input,
                                                       self.activation_output,
                                                       weighted_error_signal])

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        # Assign value inside a list so that the result (a matrix represented as a 2d array) is treated as a single
        # item and thereby assigned to a single outputState, rather than as (the usual) 2d array of items (one
        # for each outputState).
        self.value = [self.learning_signal]
        return self.value

    @property
    def activation_input(self):
        return self.variable[ACTIVATION_INPUT_INDEX]

    @activation_input.setter
    def activation_input(self, value):
        self.variable[ACTIVATION_INPUT_INDEX] = value

    @property
    def activation_output(self):
        return self.variable[ACTIVATION_OUTPUT_INDEX]

    @activation_output.setter
    def activation_output(self, value):
        self.variable[ACTIVATION_OUTPUT_INDEX] = value

    @property
    def error_signal(self):
        return self.variable[ERROR_SIGNAL_INDEX]

    @error_signal.setter
    def error_signal(self, value):
        self.variable = value