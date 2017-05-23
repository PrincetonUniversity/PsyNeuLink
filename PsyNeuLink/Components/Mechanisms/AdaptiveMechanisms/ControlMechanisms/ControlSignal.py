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

A ControlSignal is an `OutputState` specialized for use with an `EVCMechanism`. It is used to modify the
parameter of a mechanism or of its :keyword:`function` that has been
`specified for control <ControlMechanism_Control_Signals>`, in a system that regulates its performance using an
`EVCMechanism` as its `controller`.  A ControlSignal is associated with a `ControlProjection` to the `parameterState
<ParameterState>` for the parameter to be controlled.  It receives an `allocation` value specified by the
EVCMechanism's `function <EVCMechanism.function>`, and uses that to compute an `intensity` that is assigned as the
`value <ControlProjection.ControlProjection.value>` of its ControlProjection. The parameterState that receives the
ControlProjection uses that value to modify the :keyword:`value` of the mechanism's (or function's) parameter for
which it is responsible.  A ControlSignal also calculates a `cost`, based on its `intensity` and/or its time course,
that is used by the EVCMechanism to adapt its `allocation` in the future.

.. _ControlSignal_Creation:

Creating a ControlSignal
------------------------

A ControlSignal is created automatically whenever the parameter of a mechanism or of its function
is `specified for control <ControlMechanism_Control_Signals>` and the mechanism belongs to a system for which
an `EVCMechanism` is the `controller`.  Although a ControlSignal can be created using its constructor, or any of the
other ways for `creating an outputState <OutputStates_Creation>`,  this is neither necessary nor advisable,
as a ControlSignal has dedicated components and requirements for configuration that must be met for it to function
properly.

.. _ControlSignal_Structure:

Structure
---------

A ControlSignal is owned by an `EVCMechanism`, and associated with a `ControlProjection` that projects to the
`parameterState <ParameterState>` associated with the paramter to be controlled.  A ControlSignal has the following
primary attributes:

.. _ControlSignal_Allocation:

* `allocation`: assigned to the ControlSignal by the EVCMechanism to which it belongs, and converted to its
  `intensity` by its `function <ControlSignal.function>`. Its value corresponds to the current round of execution of
  the EVCMechanism to which the ControlSignal belongs.  The value in the previous round of execution can be accessed
  using the ControlSignal's `last_allocation` attribute.
..
* `allocation_samples`:  list of the allocation values to be sampled when the `EVCMechanism` to which the
  ControlSignal belongs determines its `allocation_policy <EVCMechanism.EVCMechanism.allocation_policy>`.
..
* `function <ControlSignal.function>`: converts the ControlSignal's `allocation` to its `intensity`.  By default this
  is an identity function (:keyword:`Linear(slope=1, intercept=0))`), that simpy uses the `allocation` as the
  `intensity`.  However, :keyword:`function` can be assigned another `TransferFunction`, or any other function that
  takes and returns a scalar value or 1d array.

.. _ControlSignal_Intensity:

* `intensity`:  the result of the ControlSignal`s `function <ControlSignal.function>` applied to its `allocation`,
  and used to modify the value of the parameter for which the ControlSignal is responsible.  Its value corresponds
  to the current round of execution of the EVCMechanism to which the ControlSignal belongs.  The value in the previous
  round of execution can be accessed using the ControlSignal's `lastIntensity` attribute.

.. _ControlSignal_Costs:

* *Costs*. A ControlSignal has three **cost attributes**, the values of which are calculated from its `intensity` to
  determine the total cost.  Each of these is calculated using a corresponding **cost function**.  Each of these
  functions can be customized, and the first three can be `enabled or disabled <ControlSignal_Toggle_Costs>`:

    .. _ControlSignal_Cost_Functions:

    * `intensity_cost`, calculated by the `intensity_cost_function` based on the current `intensity` of the
      ControlSignal.
    |
    * `adjustment_cost`, calculated by the `adjustment_cost_function` based on a change in the ControlSignal's
      `intensity` from its last value.
    |
    * `duration_cost`, calculated by the `duration_cost_function` based on an integral of the the ControlSignal's
      `cost`.
    |
    * `cost`, calculated by the `cost_combination_function` that combines the results of any cost functions that are
      enabled (as described in the following section).

    .. _ControlSignal_Toggle_Costs:

    *Enabling and Disabling Cost Functions*.  Any of the cost functions (except the `cost_combination_function`) can
    be enabled or disabled using the `toggle_cost_function` method to turn it `ON` or `OFF`. If it is disabled, that
    component of the cost is not included in the ControlSignal's `cost` attribute.  A cost function  can  also be
    permanently disabled for the ControlSignal by assigning it's attribute `None`.  If a cost function is permanently
    disabled for a ControlSignal, it cannot be re-enabled using `toggle_cost_function`.

  .. note:: The `index <OutputState.OutputState.index>` and `calculate <OutputState.OutputState.calculate>`
            attributes of a ControlSignal are automatically assigned and should not be modified.


.. _ControlSignal_Execution:

Execution
---------

A ControlSignal cannot be executed directly.  It is executed whenever the `EVCMechanism` to which it belongs is
executed.  When this occurs, the EVCMechanism provides the ControlSignal with an `allocation`, that is used by its
`function <ControlSignal.function>` to compute its `intensity` for that round of execution.  The `intensity` is used
by its associated `ControlProjection` to set the :keyword:`value` of the `parameterState <ParameterState>` to which it
projects. The paramemterState uses that value, in turn, to modify the value of the mechanism or function parameter
being controlled.  The ControlSignal's `intensity`is also used by its `cost functions <ControlSignal_Cost_Functions>`
to compute its `cost` attribute. That is used, along with its `allocation_samples` attribute, by the EVCMechanism to
evaluate the `expected value of control (EVC) <EVCMechanism_EVC>` of the current
`allocation_policy <EVCMechanism.EVCMechanism.allocation_policy>`, and (possibly) adjust the ControlSignal's
`allocation` for the next round of execution.

.. note::
   The changes in a parameter in response to the execution of an EVCMechanism are not applied until the mechanism
   with the parameter being controlled is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).

Class Reference
---------------

"""

# import Components
from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import *
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
    function=LinearCombination(operation=SUM),       \
    intensity_cost_function=Exponential,             \
    adjustment_cost_function=Linear,                 \
    duration_cost_function=Integrator,               \
    cost_combination_function=Reduce(operation=SUM), \
    allocation_samples=DEFAULT_ALLOCATION_SAMPLES,   \
    params=None,                                     \
    name=None,                                       \
    prefs=None)

    A subclass of OutputState that represents the output of an `EVCmechanism` provided to a `ControlProjection`.

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
        specifies the `EVCMechanism` to which to assign the ControlSignal.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to determine the `intensity` of the ControlSignal from its `allocation`.

    intensity_cost_function : Optional[TransferFuntion] : default Exponential
        specifies the function used to calculate the contribution of the ControlSignal's `intensity` to its `cost`.

    adjustment_cost_function : Optional[TransferFunction] : default Linear
        specifies the function used to calculate the contribution of the change in the ControlSignal's `intensity`
        (from its `lastIntensity` value) to its `cost`.

    duration_cost_function : Optional[IntegratorFunction] : default Integrator
        specifies the function used to calculate the contribution of the ControlSignal's duration to its `cost`.

    cost_combination_function : function : default :py:class:`Reduce(operation=SUM) <Function.Reduce>`
        speciies the function used to combine the results of any cost functions that are enabled, the result of
        whihc is assigned as the ControlSignal's `cost`.

    allocation_samples : list : default range(0.1, 1, 0.1)
        specifies the values used by `ControlSignal's `ControlSignal.owner` to determine its
        `allocation_policy <EVCMechanism.EVCMechanism.allocation_policy>` (see `ControlSignal_Execution`).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the outputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the outputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the outputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : EVCMechanism
        the `EVCMechanism` to which the ControlSignal belongs.

    allocation : float : default: defaultControlAllocation
        value used as `variable <ControlSignal.variable>` for the ControlSignal's `function <ControlSignal.function>`
        to determine its `intensity`.

    last_allocation : float
        value of `allocation` in the previous execution of ControlSignal's `owner <ControlSignal.owner>`.

    allocation_samples : list : DEFAULT_SAMPLE_VALUES
        set of values to sample by the ControlSignal's `owner <ControlSignal.owner>` to determine its
        `allocation_policy <EVCMechanism.EVCMechanism.allocation_policy>`.

    variable : number, list or np.ndarray
        same as `allocation`;  used by `function <ControlSignal.function>` to compute the ControlSignal's `intensity`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        converts `allocation` into the ControlSignal's `intensity`.  The default is the identity function, which
        assigns the ControlSignal's `allocation` as its `intensity`.

    intensity : float
        result of `function <ControlSignal.function>`;  assigned as the value of the ControlSignal's ControlProjection,
        and used to modify the value of the parameter to which the ControlSignal is assigned.

    value : number, list or np.ndarray
        result of `function <ControlSignal.function>`; same as `intensity`.

    last_intensity : float
        the `intensity` of the ControlSignal on the previous execution of its `owner <ControlSignal.owner>`.

    intensity_cost_function : TransferFunction : default default Exponential
        calculates `intensity_cost` from the curent value of `intensity`. It can be any `TransferFunction`, or any other
        function that takes and returns a scalar value. The default is `Exponential`.  It can be disabled permanently
        for the ControlSignal by assigning `None`.

    intensity_cost : float
        cost associated with the current `intensity`.

    adjustment_cost_function : TransferFunction : default Linear
        calculates `adjustment_cost` based on the change in `intensity` from  `lastIntensity`.  It can be any
        `TransferFunction`, or any other function that takes and returns a scalar value. It can be disabled
        permanently for the ControlSignal by assigning `None`.

    adjustment_cost : float
        cost associated with last change to `intensity`.

    duration_cost_function : IntegratorFunction : default Linear
        calculates an integral of the ControlSignal's `cost`.  It can be any `IntegratorFunction`, or any other
        function that takes a list or array of two values and returns a scalar value. It can be disabled permanently
        for the ControlSignal by assigning `None`.

    duration_cost : float
        intregral of `cost`.

    cost_combination_function : function : default Reduce(operation=SUM)
        combines the results of all cost functions that are enabled, and assigns the result to `cost`.
        It can be any function that takes an array and returns a scalar value.

    cost : float
        combined result of all cost functions that are enabled.

    efferents : [List[ControlProjection]]
        a list with one item -- the `ControlProjection` assigned to the ControlSignal.

    name : str : default <State subclass>-<index>
        name of the outputState.
        Specified in the **name** argument of the constructor for the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the outputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

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
        CONTROL_SIGNAL_COST_OPTIONS:ControlSignalCostOptions.DEFAULTS,
    })
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
        if params:
            try:
                if params[ALLOCATION_SAMPLES] is not None:
                    allocation_samples =  params[ALLOCATION_SAMPLES]
            except KeyError:
                pass

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

        # Validate cost functions in request_set
        #   This should be all of them if this is an initialization call;
        #   Otherwise, just those specified in assign_params
        for cost_function_name in [item for item in request_set if item in costFunctionNames]:
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
            elif isinstance(cost_function, (function_type, method_type)):
                if cost_function_name in COST_COMBINATION_FUNCTION:
                    test_value = [1, 1]
                else:
                    test_value = 1
                try:
                    result = cost_function(test_value)
                    if not (is_numeric(result) or is_numeric(np.asscalar(result))):
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
        if ALLOCATION_SAMPLES in request_set:
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

        # # If allocation_policy has been assigned, set self.value to it so it reflects the number of  controlSignals;
        # #    this is necessary, since function is not fully executed during initialization (in _instantiate_function)
        # #    it returns default_allocation policy which has only a singel item,
        # #    however validation of indices for outputStates requires that proper number of items be in self.value
        # # FIX: SHOULD VALIDATE THAT FUNCTION INDEED RETURNS A VALUE WITH LENGTH = # ControlSignals
        try:
            self.owner.value = self.owner.allocation_policy
        except AttributeError:
            pass


        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # ControlProjection Cost Functions
        for cost_function_name in [item for item in target_set if item in costFunctionNames]:
            cost_function = target_set[cost_function_name]
            if not cost_function:
                continue
            if ((not isinstance(cost_function, (Function, function_type, method_type)) and
                     not issubclass(cost_function, Function))):
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

            self.paramsCurrent[cost_function_name] = cost_function

        self.controlSignalCostOptions = self.paramsCurrent[CONTROL_SIGNAL_COST_OPTIONS]

        # Assign instance attributes
        self.allocation_samples = self.paramsCurrent[ALLOCATION_SAMPLES]

        # Default intensity params
        self.default_allocation = defaultControlAllocation
        self.allocation = self.default_allocation  # Amount of control currently licensed to this signal
        self.last_allocation = self.allocation
        self.intensity = self.allocation

        # Default cost params
        # # MODIFIED 1/23/17 OLD:
        # self.intensity_cost = self.intensityCostFunction(self.intensity)
        # MODIFIED 1/23/17 NEW:
        self.intensity_cost = self.intensity_cost_function(self.intensity)
        # MODIFIED 1/23/17 END
        self.adjustment_cost = 0
        self.duration_cost = 0
        self.last_duration_cost = self.duration_cost
        self.cost = self.intensity_cost
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
            - if ignoreIntensityFunction is set (for efficiency, if the execute method it is the identity function):
                ignore self.function
                pass allocation (input to controlSignal) along as its output
        Update cost
        Assign intensity to value of ControlSignal (done in setter property for value)

        :parameter allocation: (single item list, [0-1])
        :return: (intensity)
        """


        # MODIFIED 4/15/17 OLD: [NOT SURE WHY, BUT THIS SKIPPED OutputState.update() WHICH CALLS self.calculate()
        # super(OutputState, self).update(params=params, time_scale=time_scale, context=context)
        # MODIFIED 4/15/17 NEW: [THIS GOES THROUGH OutputState.update() WHICH CALLS self.calculate()
        super().update(params=params, time_scale=time_scale, context=context)
        # MODIFIED 4/15/17 END

        # store previous state
        self.last_allocation = self.allocation
        self.lastIntensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.duration_cost

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
            intensity_cost = self.intensity_cost = self.intensity_cost_function(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.ADJUSTMENT_COST:
            adjustment_cost = self.adjustment_cost = self.adjustment_cost_function(intensity_change)
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.DURATION_COST:
            duration_cost = self.duration_cost = self.duration_cost_function([self.last_duration_cost, new_cost])
            if self.prefs.verbosePref:
                print("++ Used duration cost")

        new_cost = self.cost_combination_function([float(intensity_cost), adjustment_cost, duration_cost])

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
                controller.log.entries[self.name + " " + kpAllocation] = LogEntry(CurrentTime(),
                                                                                  context,
                                                                                  float(self.allocation))
                controller.log.entries[self.name + " " + kpIntensityCost] =  LogEntry(CurrentTime(),
                                                                                      context,
                                                                                      float(self.intensity_cost))
                controller.log.entries[self.name + " " + kpAdjustmentCost] = LogEntry(CurrentTime(),
                                                                                      context,
                                                                                      float(self.adjustment_cost))
                controller.log.entries[self.name + " " + kpDurationCost] = LogEntry(CurrentTime(),
                                                                                    context,
                                                                                    float(self.duration_cost))
                controller.log.entries[self.name + " " + kpCost] = LogEntry(CurrentTime(),
                                                                            context,
                                                                            float(self.cost))
    #endregion

        # # MODIFIED 4/15/17 OLD: [REDUNDANT WITH ASSIGNMENT IN PROPERTY]
        # self.value = self.intensity
        # MODIFIED 4/15/17 END

    @property
    def allocation_samples(self):
        return self._allocation_samples

    @allocation_samples.setter
    def allocation_samples(self, samples):
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
        """Return three-element list with the values of ``intensity_cost``, ``adjustment_cost`` and ``duration_cost``
        """
        return [self.intensity_cost, self.adjustment_cost, self.duration_cost]



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

