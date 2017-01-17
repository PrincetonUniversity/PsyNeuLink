# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  OutputState *****************************************************

"""
# :py:data:`intensity <ControlSignal.intensity>`

Overview
--------

A ControlSignal is an :doc:`OutputState` speicialized for use with an :doc:`EVCMechanism`. It is used to modify the
parameter of a mechanism or its function that has been :ref:`specified for control <LINK>`, in a system that regulates
its performance using an :doc:`EVCMechanism` as its :ref:`controller <System.System_Base.controller>`.  A ControlSignal
is associated with a :doc:`ControlProjection` to the :doc:`parameterState <ParameterState>` for the parameter to be
controlled.  It receives an :py:data:`allocation` value specified by the EVCMechanism's ``function``, and uses that to
compute an :py:data:`intensity` that is assigned as the value of its ControlProjection. The parameterState that
receives the ControlProjection uses that value to modify the value of the mechanism's (or function's) parameter for
which it is responsible.  A ControlSignal also calculates a :py:data:`cost`, based on its intensity and/or its time
course, that is used by the EVCMechanism to adapt its allocation in the future.

.. _ControlSignal_Creation:

Creating a ControlSignal
------------------------

A ControlSignal is created automatically whenever the parameter of a mechanism or its function
:ref:` is specified for control <ControlMechanism_Specifying_Control>` and the mechanism belongs to a system for which
an :doc:`EVCMechanism` is the :py:data:`controller <System.System_Base.controller>`.  Although a
ControlSignal can be created using its constructor, or any of the other ways for
:ref:`creating an outputState  <OutputState_Creation>`,  this is neither necessary nor advisable, as a ControlSignal
has dedicated component and requirements for configuration that must be met for it to function properly.

.. _ControlSignal_Structure:

Structure
---------

A ControlSignal is owned by an :doc:`EVCMechanism`, and associated with a :doc:`ControlProjection` that projects to the
:doc:`parameterState <ParameterState>` associated with the paramter to be controlled.  Like all PsyNeuLink components,
it has the three following core attributes:

* ``variable``:  this is a 1d array that receives an :py:data:`allocation` from the EVCMechanism to which it belongs,
  and is equivalent to the ControlSignal's :py:data:`allocation`.

* ``function``: this converts the ControlSignal's :py:data:`allocation` to its :py:data:`intensity`.  By default this
  is an identity function (``(Linear(slope=1, intercept=0))``), but can be assigned another :py:doc:`TransferFunction`,
  or any other function that takes and returns a scalar value or 1d array.

* ``value``:  this is assigned the result of the ControlSignal`s ``function``, and is equivalent to the ControlSignal's
  :py:data:`intensity` attribute.

.. _ControlSignal_Cost_Attributes:

A ControlSignal also has several additional attributes and functions that determine is operation:

* :py:data:`allocation`: assigned to the ControlSignal by the EVCMechanism to which it belongs, and converted to its
  :py:data:`intensity` by its ``function``.  It is equivalent to the ControlSignal's ``value`` attribute.  The value
  corresponds to the current round of execution.  The value in the previous round of execution can be accessed using
  the ControlSignal's :py:data:`lastAllocation` attribute.

* :py:data:`intensity`: the result of the ControlSignal's ``function`` applied to its :py:data:`allocation`.  By
  default, the ControlSignal's ``function`` is an identity function that sets its :py:data:`intensity` equal to its
  :py:data:`allocation`.  The value corresponds to the current round of execution.  The value in the previous round
  of execution can be accessed using the ControlSignal's :py:data:`lastAllocation` attribute.

* :py:data:`allocation_samples`:  list of the allocation values for use by the EVCMechanism to which the
  ControlSignal belongs, when it constructs an :ref:`allocationPolicy <EVCMechanism.EVCMechanism.allocationPolicy>`
  (a particular combination of allocation values for its ControlSignals) to evaluate.

.. _ControlSignal_Costs:

* *Costs*. A ControlSignal has three **cost attributes**, the values of which are calculated from its
  :py:data:`intensity` to determine the total cost.  Each of these is calculated using a corresponding
  **cost function**.  Each of these functions can be customized, and the first three can be
  :ref:`enabled or disabled <ControlSignal_Toggle_Costs>`:

    .. _ControlSignal_Cost_Functions:

    * :py:data:`intensityCost`, calculated by the :py:data:`intensityCostFunction` based on the current
      :py:data:`intensity` of the ControlSignal.

    * :py:data:`adjustmentCost`, calculated by the :py:data:`adjustmentCostFunction` based on a change in the
      ControlSignal's :py:data:`intensity` from its last value.

    * :py:data:`durationCost`, calculated by the :py:data:`durationCostFunction` based on an integral of the
      the ControlSignal's :py:data:`cost`.

    * :py:data:`cost`, calculated by the :py:data:`costCombinationFunction` that combines the results of any cost
      functions that are enabled (as described in the following section).

    .. _ControlSignal_Toggle_Costs:

    *Enabling and Disabling Cost Functions*.  Any of the cost functions (except the
    :py:data:`cost_combination_function`) can be enabled or disabled using the :py:meth:`toggle_cost_function` method
    to turn it :keyword:`ON` or :keyword:`OFF`. If it is disabled, that component of the cost is not included in the
    ControlSignal's :py:data:`cost` attribute.  A cost function  can also be permanently disabled for the
    ControlSignal by assigning it's attribute the value of `None`.  If a cost function is permanently
    disabled for a ControlSignal, it cannot be re-enabled using :py:meth:`toggle_cost_function`.

NOTE: ControlSignals do not use the :py:data:`index <OutputState.OutputState.index>`index or
:py:data:`calculate <OutputState.OutputState.calculate>` attributes of an outputState.


.. _ControlSignal_Execution:

Execution
---------

A ControlSignal cannot be executed directly.  It is executed whenever the EVCMechanism to which it belongs is
executed.  When this occures, the EVCMechanism providesthe ControlSignal with an :py:data:`allocation`, that is used by
its ``function`` to compute its :py:data:`intensity` for that round of execution.  The :py:data:`intensity` is used
by its associated ControlProjection to set the value of the parameterState to which it projects which, in turn,
modifies the value of the mechanism or function parameter being controlled.  The :py:data:`intensity is also used by
the ControlSignal's :ref:`cost functions <ControlSignal_Cost_Functions>` to compute its :py:data:`cost` attribute.
That is used, along with its :py:data:`allocation_samples` attribute, by the EVCMechanism to evaluate
:ref:`expected value of control (EVC) <EVCMechanism_EVC>` of the current
:py:data:`allocationPolicy <EVCMechanism.EVCMechanism.allocationPolicy>`, and (possibly) adjust the ControlSignal's
:py:data:`allocation` for the next round of execution.

.. note::
   The changes in a parameter in response to the execution of an EVCMechanism are not applied until the mechanism
   with the parameter being controlled is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).

Class Reference
---------------

"""

# import Components
from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Mechanisms.ControlMechanisms.EVC.EVCMechanism import *
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.State import *


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
    OutputState(                                     \
    owner,                                           \
    value=None,                                      \
    function=LinearCombination(operation=SUM),       \
    intensity_cost_function=Exponential,             \
    adjustment_cost_function=Linear,                 \
    duration_cost_function=Integrator,               \
    cost_combination_function=Reduce(operation=SUM), \
    allocation_samples=DEFAULT_ALLOCATION_SAMPLES,   \
    params=None,                                     \
    name=None,                                       \
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

    calculate : function or method : default default :py:class:`Linear <Function.Linear>`
        used to convert item of owner mechanism's ``value`` to outputState's ``value`` (and corresponding
        item of owner's :py:data:`outputValue <Mechanism.Mechanism_Base.outputValue>`.  It must accept a value
        that has the same format (number and type of elements) as the mechanism's ``value``.

    function : Function or method : default :py:class:`LinearCombination(operation=SUM) <Function.LinearCombination>`
        function used to aggregate the values of the projections received by the outputState.
        It must produce a result that has the same format (number and type of elements) as its ``value``.
        It is implemented for consistency with other states, but is not actively used by PsyNeuLInk at the moment
        (see note under a description of the ``function`` attribute below).

    intensity_cost_function : Optional[TransferFuntion] : default :py:class:`Exponential <Function.Exponential>`
        calculates a cost based on the control signal :py:data:`intensity`.

    adjustment_cost_function : Optional[TransferFunction] : default :py:class:`Linear <Function.Linear>`
        calculates a cost based on a change in the control signal :py:data:`intensity` from its last value.

    duration_cost_function : Optional[IntegratorFunction] : default :py:class:`Integrator <Function.Integrator>`
        Calculates an integral of the ControlProjection's :py:data:`cost`.

    cost_combination_function : function : default :py:class:`Reduce(operation=SUM) <Function.Reduce>`
        Combines the results of any cost functions that are enabled, and assigns the result to :py:data:`cost`.

    allocation_samples : list : default :keyword:`DEFAULT_ALLOCATION_SAMPLES`
        List of values used by the ControlMechanism to which the ControlSignal belongs, when it constructs an
        :ref:`allocationPolicy <EVCMechanism.EVCMechanism.allocationPolicy>` (a particular combination of allocation
        values for its ControlSignals) to evaluate.  The default value is an array that ranges from 0.1 to 1 in steps
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

    index : int : default 0
        the item in the owner mechanism's ``value`` attribute used as input of the
        :py:data:`calculate <OutputState.calculate>` function, to determine the ``value`` of the outputState.

    calculate : function or method : default :py:class:`Linear <Function.Linear>`
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

    intensityCostFunction : TransferFunction : default default :py:class:`Exponential <Function.Exponential>`
        calculates "intensityCost`` from the curent value of :py:data:`intensity <ControlProjection.intensity>`.
        It can be any :py:class:`TransferFunction <Function.TransferFunction>`, or any other function  that takes and
        returns a scalar value. The default is :py:class:`Exponential <Function.Exponential>`.
        It can be disabled permanently for the ControlProjection by assigning `None`.

    adjustmentCostFunction : TransferFunction : default :py:class:`Linear <Function.Linear>`
        calculates :py:data:`adjustmentCost <ControlProjection.adjustmentCost>` based on the change in
        :py:data:`intensity <ControlProjection.intensity>` from its last value. It can be any
        :py:class:`TransferFunction <Function.TransferFunction>`, or any other function that takes and
        returns a scalar value. It can be disabled permanently for the ControlProjection by assigning `None`.

    durationCostFunction : IntegratorFunction : default :py:class:`Linear <Function.Linear>`
        calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.
        It can be any :py:class:`IntegratorFunction <Function.IntegratorFunction>`, or any other function that takes a
        list or array of two values and returns a scalar value. It can be disabled permanently for the ControlSignal by
        assigning `None`.

    costCombinationFunction : function : default :py:class:`Reduce(operation=SUM) <Function.Reduce>`
        combines the results of any cost functions that are enabled, and assigns the result to :py:data:`cost>`.
        It can be any function that takes an array and returns a scalar value.

    intensity : float
        output of ``function``, used to determine controlled parameter of task;  same as the ControlSignal's ``value``
        attribute.

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
        # In case the ControlSignal has not yet been assigned (and its value is INITIALIZING or DEFERRED_INITIALIZATION
        if isinstance(self._value, str):
            return self._value
        else:
            return self._intensity

    @value.setter
    def value(self, assignment):
        self._value = assignment

