# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  OutputState *****************************************************

"""

Overview
--------


.. _OutputStates_Creation:

Creating an OutputState
-----------------------

An outputState can be created by calling its constructor, but in general this is not necessary as a mechanism
usually creates the outputState(s) it needs automatically when it is created.  For example, if the mechanism is
being created within the :ref:`pathway of a process <Process_Pathway>`, an outputState will be created and assigned
as the ``sender`` of a MappingProjection to the next mechanism in the pathway, or to the process's
:ref:`Process_Input_And_Ouput` if it is the :keyword:`TERMINAL` mechanism of that process.

An outputState must be owned by a mechanism. Therefore, if the outputState is created directly,
the mechanism to which it belongs must be specified in the ``owner`` argument of its constructor; if the outputState
is specified in the :keyword:`OUTPUT_STATES` entry of the parameter dictionary for a mechanism, then the owner is
inferred from the context.

.. _OutputState_Primary:

Every mechanism has at least one ("primary") outputState that is automatically created, and assigned to the first
(and often only) item of the mechanism's ``value``.  The primary outputState is assigned to the mechanism's
:py:data:`outputState <Mechanism.Mechanism_Base.outputStates> attribute (note the singular), and also as the first
entry in the OrderedDictionary of the mechanism's :py:data:`outputStates <Mechanism.Mechanism_Base.outputStates>
attribute (note the plural).

In addition to the primary outputState, many mechanisms also assign an outputState for each addtional item of their
owner mechanism's ``value``, and some assign  additional outputStates that calculate values derived from one more
more of those items.  Assigning outputStates explicitly (i.e., including an :keyword:`OUTPUT_STATES` entry in the
mechanism's params dictionary) adds them to any that are automatically generated for that mechanism.  If the name of
an explicitly specified outputState is the same  as one automatically generated, the name of the former will be
suffixed with a numerical index,  and the outputState will be added to the list (that is, it will *not* replace the
one automatically generated).

.. _OutputState_Specification:

If one or more custom outputStates need to be added when a mechanism is created, or added to an existing
mechanism, they can be specified in an entry of the mechanism's parameter dictionary, using the key
:keyword:`OUTPUT_STATES` and a value that specifies the outputState for each one to be added. For a single
outputState, the value can be any of the specifications in the the list below.  To create multiple outputStates,
the value of the  :keyword:`OUTPUT_STATES` entry can be either a list, each item of which can be any of the
specifications below;  or,  it can be an OrderedDict, in which the key for each entry is a string  specifying the
name for the outputState to be  created, and its value is one of the specifications below:

    * An existing **outputState** object.  Its ``variable`` must match (in the number and type of its elements)
      the item of the owner mechanism's ``value`` to which the outputState is assigned
      (designated by its :ref:`index attribute <OutputState_Index>`).
    ..
    * The :class:`OutputState` **class** or a string.  This creates a default outputState that is assigned the first
      item of the owner mechanism's ``value`` as its value.  If :keyword:`OutputState` is used, a default name is
      assigned to the state;  if a string is, it is assigned as the name of the outputState
      (see :ref:`naming conventions <LINK>`).
    ..
    * A **specification dictionary**.  This can include entries with keys using any of the arguments in an
      outputState's constructor, and a value for that argument.  By default, the outputState is assigned to the
      first item of the owner mechanism's ``value``.  However, the :ref:`index argument <OutputState_Index>`
      can be used to assign the outputState to different item.
    ..
    * A **value**.  This creates a default outputState using the specified value as the outputState's ``variable``.
      This must be compatible with the item of the owner mechanism's ``value`` that will be assigned to the
      outputState (designated by its :ref:`index attribute <OutputState_Index>`).
      COMMENT:
         AT PRESENT THIS IS NOT USEFUL;  HOWEVER, IN THE FUTURE (E.G., WHEN GATING PROJECTIONS TO OUTPUT STATES
         IS ADDED) IT MAY BE USEFUL FOR SPECIFYING A BASEVALUE (E.G., DEFAULT) FOR THE OUTPUTSTATE.
      COMMENT

    .. note::
       In all cases, the ``variable`` of the outputState must match (have the same number and type of elements)
       as the item of its owner mechanism's ``value`` to which it is assigned.

COMMENT:
    OutputStates can also be added by using the :py:func:`assign_output_state <OutputState.assign_output_state>` method.
COMMENT

.. _OutputState_Index_and_Calculate:

By default, an explicitly specified outputState will use the first item of the owner mechanism's ``value``.  However,
it can be assigned a different item by specifying its :ref:`index <OutputState_Index>` parameter The ``variable`` of
an outputState must match (in the number and type of its elements) the item of the mechanism's ``value`` to which it
is assigned. An outputState can also be configured to transform the value of the item, by specifying a function for
its :ref:`OutputState_Calculate` parameter;  the function must be able to take as input a value that is compatible
with the item to which the outputState is assigned.

.. _OutputState_Structure:

Structure
---------


COMMENT:
    FROM ControlProjection *********************

    Overview
    --------

    A ControlProjection implements a control signal used to modify the value of a parameter of a mechanism's function.  It
    takes a value (an *allocation*) from a ControlMechanism (its ``sender``), and uses this to compute the control signal's
    :py:data:`intensity <ControlProjection.intensity>`, which is assigned as the ControlProjection's value.  The
    parameterState that receives the ControlProjection uses its value to regulate the value of a parameter of a
    mechanism's ``function``. A ControlProjection also calculates a :py:data:`cost <ControlProjection.cost>` for the
    control signal, based on its intensity  and/or its time course, that can be used by a ControlMechanism to adapt the
    ControlProjection's :py:data:`allocation <ControlProjection.allocation>`.

    .. _ControlProjection_Creation:

    Creating a ControlProjection
    ----------------------------

    A ControlProjection can be created using any of the standard ways to  :ref:`create a projection <Projection_Creation>`,
    or by including it in the :ref:`specification of a parameter <ParameterState_Specifying_Parameters>` for a mechanism,
    MappingProjection, or their ``function``.  If a ConrolProjection is created using its constructor on its own,
    the ``receiver`` argument must be specified.  If it is included in a parameter specification, the parameterState for
    the parameter being specified will be assigned as the ControlProjection's ``receiver``.  If its ``sender`` is not
    specified, its assignment depends on the ``receiver``.  If the receiver belongs to a mechanism that is part of a
    system, then the ControlProjection's ``sender`` is assigned to an outputState of the system's
    :ref:`controller  <System_Execution_Control>`. Otherwise, the ``sender`` is assigned to the outputState of a
    :any:`DefaultControlMechanism`.

    The four functions used to calculate the :ref:`cost of a control signal <ControlProjection_Cost_Functions>`
    can be specified  either in arguments to the ControlProjection's constructor, or in a ``params dictionary`` (see
    :ref:`ControlProjection_Cost_Functions` below). A custom function can be assigned to any cost function, so long as
    it accepts the appropriate type of value and returns a scalar.  Each of the cost functions can be
    :ref:`enabled or disabled <ControlProjection_Toggle_Costs>`, to select which make contributions to the
    ControlProjection's :py:data:`cost <ControlProjection.cost>`.  A cost function can also be permanently disabled for
    its ControlProjection by assigning :keyword:`None` to the argument for that function in its constructor (or the
    appropriate entry in its ``params`` dictionary). Cost functions that are permanently disabled in this way cannot be
    re-enabled.

    A ControlProjection takes an ``allocation_samples`` specification as its input.  This must be an array that
    specifies the values of its :py:data:`allocation <ControlProjection.allocation>` that will be sampled by
    ControlMechanisms that adaptively adjust ControlProjection allocations (e.g., :doc:`EVCMechanism`).  The default is
    an array of values from 0.1 to 1.0 in steps of 0.1.

    .. _ControlProjection_Structure:

    Structure
    ---------

    *Intensity*. The ControlProjection's ``function`` uses its :py:data:`allocation <ControlProjection.allocation>` to
    calculate a control signal :py:data:`intensity <ControlProjection.intensity>`. The default is an identity function
    ``(Linear(slope=1, intercept=0))``: the ControlProjection sets its control signal
    :py:data:`intensity <ControlProjection.intensity>` equal to its :py:data:`allocation <ControlProjection.allocation>`.
    The ``function`` can be assigned another :py:doc:`TransferFunction`, or any other function that takes and returns a
    scalar value.

    *Costs*. A ControlProjection has four cost functions that determine how the ControlProjection computes the cost of
    its control signal, all of which can be customized, and the first three of which can be enabled or disabled:

    .. _ControlProjection_Cost_Functions:

    * :keyword:`INTENSTITY_COST_FUNCTION`
        Calculates a cost based on the control signal :py:data:`intensity <ControlProjection.intensity>`.
        It can be any :class:`TransferFunction`, or any other function  that takes and returns a scalar value.
        The default is :class:`Exponential`.

    * :keyword:`ADJUSTMENT_COST_FUNCTION`
        Calculates a cost based on a change in the control signal :py:data:`intensity <ControlProjection.intensity>`
        from its last value. It can be any :class:`TransferFunction`, or any other function that takes and returns a
        scalar value. The default is :py:class:`Function.Linear`.

    * :keyword:`DURATION_COST_FUNCTION`
        Calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.
        It can be any :class:`IntegratorFunction`, or any other function  that takes a list or array of two values and
        returns a scalar value. The default is :class:`Integrator`.

    * :keyword:`COST_COMBINATION_FUNCTION`
        Combines the results of any cost functions that are enabled, and assigns the result as the ControlProjection's
        :py:data:`cost <ControlProjection.cost>`.  It can be any function that takes an array and returns a scalar value.
        The default is :py:class:`Reduce`.

    An attribute is assigned for each component of the cost
    (:py:data:`intensityCost <ControlProjection.intensityCost>`,
    :py:data:`adjustmentCost <ControlProjection.adjustmentCost>`, and
    :py:data:`durationCost <ControlProjection.durationCost>`),
    and the total cost (:py:data:`cost <ControlProjection.cost>`.

    .. _ControlProjection_Toggle_Costs:

    *Toggling Cost Functions*.  Any of the cost functions (except the :keyword:`COST_COMBINATION_FUNCTION`) can be
    enabled or disabled using the :py:meth:`toggle_cost_function <ControlProjection.toggle_cost_function>` method
    to turn it :keyword:`ON` or :keyword:`OFF`.  If it is disabled, that component of the cost is not included in the
    ControlProjection's :py:data:`cost <ControlProjection.cost>` attribute.  A cost function  can also be permanently
    disabled for the ControlProjection by assigning :keyword:`None` to its argument in the constructor (or the
    corresponding entry in its ``params`` dictionary).  If a cost function is permanently disabled for a ControlProjection,
    it cannot be re-enabled using :py:meth:`toggle_cost_function <ControlProjection.toggle_cost_function>`.

    *Additional Attributes*.  In addition to the intensity and cost attributes described above, a ControlProjection has
    :py:data:`lastAllocation <ControlProjection.lastAllocation>` and
    :py:data:`lastIntensity <ControlProjection.lastIntensity>` attributes that store the values associated with its
    previous execution. Finally, it has an :py:data:`allocation_samples <ControlProjection.allocation_samples>` attribute,
    that is a  list of used by :ref:`ControlMechanisms  <ControlMechanism>` for sampling different values of
    :py:data:`allocation <ControlProjection.allocation>` for the ControlProjection, in order to adaptively adjust the
    parameters that it controls (e.g., :doc:`EVCMechanism`). The default value is an array that ranges from
    0.1 to 1.0 in steps of 0.1.

    .. _ControlProjection_Execution:

    Execution
    ---------

    A ControlProjection uses its ``function`` to compute the :py:data:`intensity <ControlProjection.intensity>` of its
    control signal, and its :ref:`cost functions <ControlProjection_Cost_Functions> use that to compute the its
    :py:data:`cost <ControlProjection.cost>`.  The :py:data:`intensity <ControlProjection.intensity>` is assigned to the
    ControlProjection's ``value`` attribute, which is used by the parmaterState to which it projects to modify the
    corresponding parameter of the owner mechanism's function.  The :py:data:`cost <ControlProjection.cost>` is used by
    the :doc:`EVCMechanism` to determine the ControlProjection's :py:data:`allocation <ControlProjection.allocation>`
    in future executions.

    .. note::
       The changes in a parameter in response to the execution of a ControlProjection are not applied until the
       mechanism that receives the projection are next executed; see Lazy_Evaluation for an explanation of "lazy"
       updating).

     *********************************************

COMMENT



Every outputState is owned by a :doc:`mechanism <Mechanism>`. It can send one or more MappingProjections to other
mechanisms;  it can also  be treated as the output of a process or system to which its owner belongs (if it is the
:keyword:`TERMINAL` mechanism for that process or system -- see :ref:`Process_Input_And_Ouput`).  A list of projections
sent by an outputState is maintained in its :py:data:`sendsToProjections <OutputState.sendsToProjections>` attribute.

Like all PsyNeuLink components, it has the three following core attributes:

* ``variable``:  this must match (both in number and types of elements) the value of the item of its owner mechanism's
  ``value`` attribute to which it is assigned (in its :py:data:`index <OutputState.index>` attribute).

* ``function``: this is implemented for potential future use, but is not actively used by PsyNeuLink at the moment.

* ``value``:  this is assigned the result of the outputState`s ``function``, possibly modifed by its
  :py:data:`calculate <OutputState.calculate>` parameter, and used as the input to any projections that it sends.

.. _OutputState_Attributes:

An outputState also has two additional attributes that determine its operation:

.. _OutputState_Index:

* :py:data:`index <OutputState.index>`: this determines the item of its owner mechanism's ``value`` to which it is
  assigned.  By default, this is set to 0, which assigns it to the first item.

.. _OutputState_Calculate:

* :py:data:`calculate <OutputState.calculate>`:  this specifies the function used to convert the item to which the
  outputState is assigned to the outputState's value.  The result is assigned to the outputState's ``value``
  attribute. The default for :py:data:`calculate <OutputState.calculate>` is the identity function, which simply
  assigns the item of the mechanism'sm ``value`` unmodified as the ``value`` of the outputState.  However,
  it can be assigned any function that can take as input the  value of the item to which the outputState is assigned.
  Note that the :py:data:`calculate <OutputState.calculate>` function is distinct from the outputState's ``function``
  parameter (which is reserved for future use).

.. _OutputState_Execution:

Execution
---------

An outputState cannot be executed directly.  It is executed when the mechanism to which it belongs is executed.
When this occurs, the mechanism places the results of its execution in its ``value`` attribute, and the value of the
outputState is then updated by calling its :py:data:`calculate <OutputState.calculate>` function using as its input
the item of the onwer mechanism's ``value`` to which the outputState is assigned.  The result is assigned to the
outputState's ``value``, as well as to a corresponding item of the mechanism's
:py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` attribute.  It is also used as the input to any
projections for which the outputState is the sender.

.. _OutputState_Class_Reference:

Class Reference
---------------


"""

# import Components
from PsyNeuLink.Components.Mechanisms.ControlMechanisms.EVCMechanism import *
from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state_list
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.Functions.Function import *

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

PRIMARY_OUTPUT_STATE = 0

class OutputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ControlSignal(OutputState):
    """
    OutputState(                               \
    owner,                                     \
    value=None,                                \
    function=LinearCombination(operation=SUM), \
    intensity_cost_function=Exponential,             \
    adjustment_cost_function=Linear,                 \
    duration_cost_function=Integrator,               \
    cost_combination_function=Reduce(operation=SUM), \
    allocation_samples=DEFAULT_ALLOCATION_SAMPLES,   \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Implements subclass of State that represents the output of a mechanism

    COMMENT:

        Description
        -----------
            The OutputState class is a type in the State category of Component,
            It is used primarily as the sender for MappingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = OUTPUT_STATES
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS   (Operation.PRODUCT)
            + paramNames (dict)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: LinearCombination with Operation.SUM)

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the mechanism to which the outputState belongs; it must be specified or determinable from the context in which
        the outputState is created.

    reference_value : number, list or np.ndarray
        a template for the item of the owner mechanism's ``value`` attribute to which the outputState will be assigned
        (specified by the ``index`` argument).  This must match (in number and type of elements) the ``variable``
        argument; it is used to insure the compatibility of the source of the input to the outputState and its
        ``variable`` (used for its ``function`` and ``calculate`` routines).

    value : number, list or np.ndarray
        used as the template for ``variable``.

    index : int : default PRIMARY_OUTPUT_STATE
        the item in the owner mechanism's ``value`` attribute used as input of the
        :py:data:`calculate <OutputState.calculate>` function, to determine the ``value`` of the outputState.

    calculate : function or method : default Linear
        used to convert item of owner mechanism's ``value`` to outputState's ``value`` (and corresponding
        item of owner's :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>`.  It must accept a value
        that has the same format (number and type of elements) as the mechanism's ``value``.

    function : Function or method : default LinearCombination(operation=SUM)
        function used to aggregate the values of the projections received by the outputState.
        It must produce a result that has the same format (number and type of elements) as its ``value``.
        It is implemented for consistency with other states, but is not actively used by PsyNeuLInk at the moment
        (see note under a description of the ``function`` attribute below).

    intensity_cost_function : Optional[TransferFuntion] : default Exponential
        calculates a cost based on the control signal :py:data:`intensity <ControlProjection.intensity>`.
        It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    adjustment_cost_function : Optiona[TransferFunction] : default Linear
        calculates a cost based on a change in the control signal :py:data:`intensity <ControlProjection.intensity>`
        from its last value. It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    duration_cost_function : Optional[TransferFunction] : default Linear
        Calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.
        It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    cost_combination_function : function : default Reduce(operation=SUM)
        Combines the results of any cost functions that are enabled, and assigns the result to
        :py:data:`cost <ControlProjection.cost>`.

    allocation_samples : list : default :keyword:`DEFAULT_ALLOCATION_SAMPLES`
        Set of values used by ControlMechanisms that sample different allocation values in order to adaptively adjust
        the function of mechanisms in their systems.  The default value is an array that ranges from 0.1 to 1 in steps
        of 0.1.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the outputState, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Component` for specification of a params dict).

    name : str : default OutputState-<index>
        a string used for the name of the outputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the outputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the PreferenceSet for the outputState.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism
        the mechanism to which the outputState belongs.

    sendsToProjections : Optional[List[Projection]]
        a list of the projections sent by the outputState (i.e., for which the outputState is a ``sender``).

    variable : number, list or np.ndarray
        assigned an item of the :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>` of its owner mechanism.

    index : int
        the item in the owner mechanism's ``value`` attribute used as input of the
        :py:data:`calculate <OutputState.calculate>` function, to determine the ``value`` of the outputState.

    calculate : function or method : default Linear
        function used to convert the item of owner mechanism's ``value`` specified by the
        :py:data:`index <OutputState.index>` attribute;  it is combined with the result of the outputState's
        ``function`` to determine it's ``value``, and the corresponding item of the owner mechanism's
        :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>`. Default is Linear (identity function)
        which simply transfers the value as is.

    value : number, list or np.ndarray
        assigned the result of the :py:data:`calculate <OutputState.calculate>` function, combined with any result of
        the outputState's ``function``, which is also assigned to the corresopnding item of the owner mechanism's
        :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>`.

    COMMENT:
        ControlSignal_State_Attributes:
    COMMENT

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation  of the ``values`` of the projections received by the
        outputState.  The result is combined with the result of the calculate function and assigned as the ``value``
        of the outputState, and the corresponding item of the owner's
        :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>`.

        .. note::
           Currently PsyNeuLink does not support projections to outputStates.  The ``function`` attribute is
           implemented for consistency with other states classes, and for potential future use.  The default simply
           passes its input to its output. The ``function`` attribute can be modified to change this behavior.
           However, for compatibility with future versions, it is *strongly* recommended that such functionality
           be implemented by assigning the desired function to the :py:data:`calculate <OutputState.calculate>`
           attribute; this will insure compatibility with future versions.

    allocation : float : default: defaultControlAllocation
        value used as ``variable`` for ControlProjection's ``function`` to determine its control signal
        :py:data:`intensity <ControlProjection.intensity>`.

    allocationSamples : list : DEFAULT_SAMPLE_VALUES
        set of values used by ControlMechanisms that sample different allocation values in order to
        adaptively adjust the function of mechanisms in their systems.

        .. _ControlSignal_Function_Attributes:

    function : TransferFunction :  default Linear
        converts :py:data:`allocation <ControlProjection.allocation>` into `control signal
        :py:data:`intensity <ControlProjection.intensity>` that is provided as output to receiver of projection.

    intensityCostFunction : TransferFunction : default Exponential
        calculates "intensityCost`` from the curent value of :py:data:`intensity <ControlProjection.intensity>`.

    adjustmentCostFunction : TransferFunction : default Linear
        calculates :py:data:`adjustmentCost <ControlProjection.adjustmentCost>` based on the change in
        :py:data:`intensity <ControlProjection.intensity>` from its last value.

    durationCostFunction : IntegratorFunction : default Linear
        calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.

    costCombinationFunction : function : default Reduce(operation=SUM)
        combines the results of any cost functions that are enabled, and assigns the result to
        :py:data:`cost <ControlProjection.cost>`.

    intensity : float
        output of ``function``, used to determine controlled parameter of task.

    intensityCost : float
        cost associated with current :py:data:`intensity <ControlProjection.intensity>`.

    adjustmentCost : float
        cost associated with last change to :py:data:`intensity <ControlProjection.intensity>`.

    durationCost
        intregral of :py:data:`cost <ControlProjection.cost>`.

    cost : float
        current value of ControlProjection's :py:data:`cost <ControlProjection.cost>`;
        combined result of all cost functions that are enabled.

    COMMENT:
        ControlSignal_History_Attributes:
    COMMENT

    lastAllocation : float
        :py:data:`allocation <ControlProjection.allocation>` for last execution of the ControlProjection.

    lastIntensity : float
        :py:data:`intensity <ControlProjection.intensity>` for last execution of the ControlProjection.

        .. _ControlProjection_Cost_Functions:

    name : str : default <State subclass>-<index>
        name of the outputState.
        Specified in the name argument of the call to create the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the PreferenceSet for the outputState.
        Specified in the prefs argument of the call to create the projection;  if it is not specified, a default is
        assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATES
    paramsType = OUTPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TYPE: MAPPING_PROJECTION,
        CONTROL_SIGNAL_COST_OPTIONS:ControlSignalCostOptions.DEFAULTS})
    #endregion


    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value,
                 variable=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 intensity_cost_function:(is_function_type)=Exponential,
                 adjustment_cost_function:tc.optional(is_function_type)=Linear,
                 duration_cost_function:tc.optional(is_function_type)=Integrator,
                 cost_combination_function:tc.optional(is_function_type)=Reduce(operation=SUM),
                 allocation_samples=DEFAULT_ALLOCATION_SAMPLES,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Note index and calculate are not used by ControlSignal, but included here for consistency with OutputState

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  intensity_cost_function=intensity_cost_function,
                                                  adjustment_cost_function=adjustment_cost_function,
                                                  duration_cost_function=duration_cost_function,
                                                  cost_combination_function=cost_combination_function,
                                                  allocation_samples=allocation_samples,
                                                  params=params)

        self.reference_value = reference_value

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         reference_value,
                         variable=variable,
                         index=index,
                         calculate=calculate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def _validate_variable(self, variable, context=None):
        """Insure variable is compatible with output component of owner.function relevant to this state

        Validate self.variable against component of owner's value (output of owner's function)
             that corresponds to this outputState (since that is what is used as the input to OutputState);
             this should have been provided as reference_value in the call to OutputState__init__()

        Note:
        * This method is called only if the parameterValidationPref is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """
        super(OutputState,self)._validate_variable(variable, context)

        self.variableClassDefault = self.reference_value

        # Insure that self.variable is compatible with (relevant item of) output value of owner's function
        if not iscompatible(self.variable, self.reference_value):
            raise OutputStateError("Value ({0}) of outputState for {1} is not compatible with "
                                           "the output ({2}) of its function".
                                           format(self.value,
                                                  self.owner.name,
                                                  self.reference_value))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate allocation_samples and controlSignal cost functions

        Checks if:
        - cost functions are all appropriate
        - allocation_samples is a list with 2 numbers
        - all cost functions are references to valid ControlProjection costFunctions (listed in self.costFunctions)
        - IntensityFunction is identity function, in which case ignoreIntensityFunction flag is set (for efficiency)

        """

        # Validate cost functions:
        for cost_function_name in costFunctionNames:
            cost_function = request_set[cost_function_name]

            # cost function assigned None: OK
            if not cost_function:
                continue

            # cost_function is Function class specification:
            #    instantiate it and test below
            if inspect.isclass(cost_function) and issubclass(cost_function, Function):
                cost_function = cost_function()

            # cost_function is Function object:
            #     COST_COMBINATION_FUNCTION must be CombinationFunction
            #     DURATION_COST_FUNCTION must be an IntegratorFunction
            #     others must be TransferFunction
            if isinstance(cost_function, Function):
                if cost_function_name == COST_COMBINATION_FUNCTION:
                    if not isinstance(cost_function, CombinationFunction):
                        raise ControlSignalError("Assignment of Function to {} ({}) must be a CombinationFunction".
                                                 format(COST_COMBINATION_FUNCTION, cost_function))
                elif cost_function_name == DURATION_COST_FUNCTION:
                    if not isinstance(cost_function, IntegratorFunction):
                        raise ControlSignalError("Assignment of Function to {} ({}) must be an IntegratorFunction".
                                                 format(DURATION_COST_FUNCTION, cost_function))
                elif not isinstance(cost_function, TransferFunction):
                    raise ControlSignalError("Assignment of Function to {} ({}) must be a TransferFunction".
                                             format(cost_function_name, cost_function))

            # cost_function is custom-specified function
            #     DURATION_COST_FUNCTION and COST_COMBINATION_FUNCTION must accept an array
            #     others must accept a scalar
            #     all must return a scalar
            elif isinstance(cost_function, function_type):
                if cost_function_name in {DURATION_COST_FUNCTION, COST_COMBINATION_FUNCTION}:
                    test_value = [1, 1]
                else:
                    test_value = 1
                try:
                    if not is_numeric(cost_function(test_value)):
                        raise ControlSignalError("Function assigned to {} ({}) must return a scalar".
                                                 format(cost_function_name, cost_function))
                except:
                    raise ControlSignalError("Function assigned to {} ({}) must accept {}".
                                             format(cost_function_name, cost_function, type(test_value)))

            # Unrecognized function assignment
            else:
                raise ControlSignalError("Unrecognized function ({}) assigned to {}".
                                         format(cost_function, cost_function_name))

        # Validate allocation samples list:
        # - default is 1D np.array (defined by DEFAULT_ALLOCATION_SAMPLES)
        # - however, for convenience and compatibility, allow lists:
        #    check if it is a list of numbers, and if so convert to np.array
        allocation_samples = request_set[ALLOCATION_SAMPLES]
        if isinstance(allocation_samples, list):
            if iscompatible(allocation_samples, **{kwCompatibilityType: list,
                                                       kwCompatibilityNumeric: True,
                                                       kwCompatibilityLength: False,
                                                       }):
                # Convert to np.array to be compatible with default value
                request_set[ALLOCATION_SAMPLES] = np.array(allocation_samples)
        elif isinstance(allocation_samples, np.ndarray) and allocation_samples.ndim == 1:
            pass
        else:
            raise ControlSignalError("allocation_samples argument ({}) in {} must be "
                                         "a list or 1D np.array of numbers".
                                     format(allocation_samples, self.name))

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # ControlProjection Cost Functions
        for cost_function_name in costFunctionNames:
            cost_function = target_set[cost_function_name]
            if not cost_function:
                continue
            if (not isinstance(cost_function, (Function, function_type)) and not issubclass(cost_function, Function)):
                raise ControlSignalError("{0} not a valid Function".format(cost_function))

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        # Instantiate cost functions (if necessary) and assign to attributes
        for cost_function_name in costFunctionNames:
            cost_function = self.paramsCurrent[cost_function_name]
            # cost function assigned None
            if not cost_function:
                self.toggle_cost_function(cost_function_name, OFF)
                continue
            # cost_function is Function class specification
            if inspect.isclass(cost_function) and issubclass(cost_function, Function):
                cost_function = cost_function()
            # cost_function is Function object
            if isinstance(cost_function, Function):
                cost_function.owner = self
                cost_function = cost_function.function
            # cost_function is custom-specified function
            elif isinstance(cost_function, function_type):
                pass
            # safeguard/sanity check (should never happen if validation is working properly)
            else:
                raise ControlSignalError("{} is not a valid cost function for {}".
                                         format(cost_function, cost_function_name))

            setattr(self,  underscore_to_camelCase('_'+cost_function_name), cost_function)

        self.controlSignalCostOptions = self.paramsCurrent[CONTROL_SIGNAL_COST_OPTIONS]

        # Assign instance attributes
        self.allocationSamples = self.paramsCurrent[ALLOCATION_SAMPLES]

        # Default intensity params
        self.default_allocation = defaultControlAllocation
        self.allocation = self.default_allocation  # Amount of control currently licensed to this signal
        self.lastAllocation = self.allocation
        self.intensity = self.allocation

        # Default cost params
        self.intensityCost = self.intensityCostFunction(self.intensity)
        self.adjustmentCost = 0
        self.durationCost = 0
        self.last_duration_cost = self.durationCost
        self.cost = self.intensityCost
        self.last_cost = self.cost

        # If intensity function (self.function) is identity function, set ignoreIntensityFunction
        function = self.params[FUNCTION]
        function_params = self.params[FUNCTION_PARAMS]
        if ((isinstance(function, Linear) or (inspect.isclass(function) and issubclass(function, Linear)) and
                function_params[SLOPE] == 1 and
                function_params[INTERCEPT] == 0)):
            self.ignoreIntensityFunction = True
        else:
            self.ignoreIntensityFunction = False

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate calculate function
        """
        super()._instantiate_attributes_after_function(context=context)

        self.intensity = self.function(self.allocation)
        self.lastIntensity = self.intensity

    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Adjust the control signal, based on the allocation value passed to it

        Computes new intensity and cost attributes from allocation

        Use self.function to assign intensity
            - if ignoreIntensityFunction is set (for effiency, if the the execute method it is the identity function):
                ignore self.function
                pass allocation (input to controlSignal) along as its output
        Update cost

        :parameter allocation: (single item list, [0-1])
        :return: (intensity)
        """

        super(OutputState, self).update(params=params, time_scale=time_scale, context=context)

        # store previous state
        self.lastAllocation = self.allocation
        self.lastIntensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.durationCost

        # update current intensity
        # FIX: INDEX MUST BE ASSIGNED WHEN OUTPUTSTATE IS CREATED FOR ControlMechanism (IN PLACE OF LIST OF PROJECTIONS)
        self.allocation = self.owner.value[self.index]
        # self.allocation = self.sender.value

        if self.ignoreIntensityFunction:
            # self.set_intensity(self.allocation)
            self.intensity = self.allocation
        else:
            self.intensity = self.function(self.allocation, params)
        intensity_change = self.intensity-self.lastIntensity

        if self.prefs.verbosePref:
            intensity_change_string = "no change"
            if intensity_change < 0:
                intensity_change_string = str(intensity_change)
            elif intensity_change > 0:
                intensity_change_string = "+" + str(intensity_change)
            if self.prefs.verbosePref:
                warnings.warn("\nIntensity: {0} [{1}] (for allocation {2})".format(self.intensity,
                                                                                   intensity_change_string,
                                                                                   self.allocation))
                warnings.warn("[Intensity function {0}]".format(["ignored", "used"][self.ignoreIntensityFunction]))

        # compute cost(s)
        new_cost = intensity_cost = adjustment_cost = duration_cost = 0

        if self.controlSignalCostOptions & ControlSignalCostOptions.INTENSITY_COST:
            intensity_cost = self.intensityCost = self.intensityCostFunction(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.ADJUSTMENT_COST:
            adjustment_cost = self.adjustmentCost = self.adjustmentCostFunction(intensity_change)
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.DURATION_COST:
            duration_cost = self.durationCost = self.durationCostFunction([self.last_duration_cost, new_cost])
            if self.prefs.verbosePref:
                print("++ Used duration cost")

        new_cost = self.costCombinationFunction([float(intensity_cost), adjustment_cost, duration_cost])

        if new_cost < 0:
            new_cost = 0
        self.cost = new_cost


        # Report new values to stdio
        if self.prefs.verbosePref:
            cost_change = new_cost - self.last_cost
            cost_change_string = "no change"
            if cost_change < 0:
                cost_change_string = str(cost_change)
            elif cost_change > 0:
                cost_change_string = "+" + str(cost_change)
            print("Cost: {0} [{1}])".format(self.cost, cost_change_string))

        #region Record controlSignal values in owner mechanism's log
        # Notes:
        # * Log controlSignals for ALL states of a given mechanism in the mechanism's log
        # * Log controlSignals for EACH state in a separate entry of the mechanism's log

        # Get receiver mechanism and state
        controller = self.owner

        # Get logPref for mechanism
        log_pref = controller.prefs.logPref

        # Get context
        if not context:
            context = controller.name + " " + self.name + kwAssign
        else:
            context = context + SEPARATOR_BAR + self.name + kwAssign

        # If context is consistent with log_pref:
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context))):
            # record info in log

# FIX: ENCODE ALL OF THIS AS 1D ARRAYS IN 2D PROJECTION VALUE, AND PASS TO .value FOR LOGGING
            controller.log.entries[self.name + " " +
                                      kpIntensity] = LogEntry(CurrentTime(), context, float(self.intensity))
            if not self.ignoreIntensityFunction:
                controller.log.entries[self.name + " " + kpAllocation] =     \
                    LogEntry(CurrentTime(), context, float(self.allocation))
                controller.log.entries[self.name + " " + kpIntensityCost] =  \
                    LogEntry(CurrentTime(), context, float(self.intensityCost))
                controller.log.entries[self.name + " " + kpAdjustmentCost] = \
                    LogEntry(CurrentTime(), context, float(self.adjustmentCost))
                controller.log.entries[self.name + " " + kpDurationCost] =   \
                    LogEntry(CurrentTime(), context, float(self.durationCost))
                controller.log.entries[self.name + " " + kpCost] =           \
                    LogEntry(CurrentTime(), context, float(self.cost))
    #endregion

        self.value = self.intensity

    @property
    def allocationSamples(self):
        return self._allocation_samples

    @allocationSamples.setter
    def allocationSamples(self, samples):
        if isinstance(samples, (list, np.ndarray)):
            self._allocation_samples = list(samples)
            return
        if isinstance(samples, tuple):
            self._allocation_samples = samples
            sample_range = samples
        elif samples == AUTO:
            # THIS IS A STUB, TO BE REPLACED BY AN ACTUAL COMPUTATION OF THE ALLOCATION RANGE
            raise ControlSignalError("AUTO not yet supported for {} param of ControlProjection; default will be used".
                                     format(ALLOCATION_SAMPLES))
        else:
            sample_range = DEFAULT_ALLOCATION_SAMPLES
        self._allocation_samples = []
        i = sample_range[0]
        while i < sample_range[1]:
            self._allocation_samples.append(i)
            i += sample_range[2]

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, new_value):
        try:
            old_value = self._intensity
        except AttributeError:
            old_value = 0
        self._intensity = new_value
        # if len(self.observers[kpIntensity]):
        #     for observer in self.observers[kpIntensity]:
        #         observer.observe_value_at_keypath(kpIntensity, old_value, new_value)

    def toggle_cost_function(self, cost_function_name, assignment=ON):
        """Enables/disables use of a cost function.

        ``cost_function_name`` should be a keyword (list under :ref:`Structure <ControlProjection_Structure>`).
        """

        if cost_function_name == INTENSITY_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.INTENSITY_COST
        elif cost_function_name == DURATION_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.DURATION_COST
        elif cost_function_name == ADJUSTMENT_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.ADJUSTMENT_COST
        elif cost_function_name == COST_COMBINATION_FUNCTION:
            raise ControlSignalError("{} cannot be disabled".format(COST_COMBINATION_FUNCTION))
        else:
            raise ControlSignalError("toggle_cost_function: unrecognized cost function: {}".format(cost_function_name))

        if assignment:
            if not self.paramsCurrent[cost_function_name]:
                raise ControlSignalError("Unable to toggle {} ON as function assignment is \'None\'".
                                         format(cost_function_name))
            self.controlSignalCostOptions |= cost_option
        else:
            self.controlSignalCostOptions &= ~cost_option

    # def set_intensity_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.INTENSITY_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.INTENSITY_COST
    #
    # def set_adjustment_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.ADJUSTMENT_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.ADJUSTMENT_COST
    #
    # def set_duration_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.DURATION_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.DURATION_COST
    #
    def get_costs(self):
        """Return three-element list with the values of ``intensityCost``, ``adjustmentCost`` and ``durationCost``
        """
        return [self.intensityCost, self.adjustmentCost, self.durationCost]



    @property
    def value(self):
        if isinstance(self._value, str):
            return self._value
        else:
            return self._intensity

    @value.setter
    def value(self, assignment):
        self._value = assignment

