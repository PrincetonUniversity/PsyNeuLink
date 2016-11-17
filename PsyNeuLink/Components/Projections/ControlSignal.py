# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ControlSignal *********************************************************

"""
.. _ControlSignal_Overview:

Overview
--------

A ControlSignal projection takes a value (an *allocation*) from a ControlMechanism (its ``sender``), and uses this to
compute its ``intensity`` that is assigned as the ControlSignal's value.  Its value is used to modify the value of a
parameterState (its ''receiver'') associated with the parameter of a function of a ProcessingMechanism.  A
ControlSignal also has an associated ``cost`` that is calculated based on its intensity and/or its time course.

.. _ControlSignal_Creating_A_ControlSignal_Projection:

Creating a ControlSignal Projection
-----------------------------------

A ControlSignal projection can be created in any of the ways that can be used to
:ref:`create a projection <Projection_Creating_A_Projection>`, or by including it in the specification for the
:ref:`parameter of a mechanism's function <Mechanism_Assigning_A_Control_Signal>`.  If the constructor is used,
the ``receiver`` argument must be specified.  If it is included in a parameter specification, its ``receiver`` will be
assigned to the parameterState for the parameter.  If its ``sender`` is not specified, its assignment depends on
the ``receiver``.  If the receiver belongs to a mechanism that is part of a system, then the ControlSignal's
``sender`` is assigned to an outputState of the system's :ref:`controller <System_Execution_Control>`.
Otherwise, the ``sender`` is assigned to the outputState of a :doc:`DefaultControlMechanism`.

The cost of a ControlSignal is calculated from its ``intensity``, using four
:ref:`cost functions <ControlSignal_Cost_Functions>` that can be specified  either in arguments to its constructor,
or in a params dictionary[LINK](see below [LINK]).  A custom function can be assigned to any cost function,
so long as it accepts the appropriate type of value (see below [LINK]) and returns a scalar.  Each of the cost
functions can be :ref:`enabled or disabled <ControlSignal_Toggle_Costs>`, to select which make contributions to the
ControlSignal's ``cost``.  A cost function can also be permanently disabled for its ControlSignal by assigning
``None`` to the argument for that function in its constructor (or the appropriate entry in its params dictionary).
Cost functions that are permanently disabled in this way cannot be re-enabled.

A ControlSignal projection takes an ``allocation_samples`` specification as its input.  This must be an array that
specifies the values of its ``allocation`` that will be sampled by ControlMechanisms that adaptively adjust
ControlSignal allocations (e.g., :doc:`EVCMechanism`[LINK]).  The default is an array of values from 0.1 to 1.0 in
steps of 0.1.

.. _ControlSignal_Structure:

Structure
---------

The ControlSignal's ``function`` calculates its ``intensity`` from its ``allocation``.  The default is an identity
function (Linear(slope=1, intercept=0)), and the ControlSignal's ``intensity`` is equal to its ``allocation``.  The
``function`` can assigned another :class:`TransferFunction`, or any other function that takes and returns a scalar
value.  In addition, there are four functions that determine how the ControlSignal computes its cost, all of which
can be customized, and the first three of which can be enabled or disabled:

.. _ControlSignal_Cost_Functions:

* :keyword:`INTENSTITY_COST_FUNCTION`
    Calculates a cost based on the ControlSignal's ``intensity``.
    It can be any :class:`TransferFunction`, or any other function  that takes and returns a scalar value.
    The default is :class:`Exponential`.

* :keyword:`ADJUSTMENT_COST_FUNCTION`
    Calculates a cost based on a change in the ControlSignal's ``intensity`` from its last value.
    It can be any :class:`TransferFunction`, or any other function that takes and returns a scalar value.
    The default is :class:`Linear`.

* :keyword:`DURATION_COST_FUNCTION`
    Calculates an integral of the ControlSignal's ``cost``.
    It can be any :class:`IntegratorFunction`, or any other function  that takes a list or array of two values and
    returns a scalar value. The default is :class:`Integrator`.

* :keyword:`COST_COMBINATION_FUNCTION`
    Combines the results of any cost functions that are enabled, and assigns the result as the ControlSignal's
    ``cost``.  It can be any :class:`CombinationFunction`, or any other function that takes an array and returns a
    scalar value.  The default is :class:`LinearCombination`.

.. _ControlSignal_Toggle_Costs:

Any of the cost functions (except the :keyword:`COST_COMBINATION_FUNCTION`) can be enabled or disabled using the
``toggle_cost_function`` method to turn it :keyword:`ON` or :keyword:`OFF`.  If it is disabled, that component of the
cost is not included in the ControlSignal's ``cost`` attribute. A cost function  can also be permanently disabled for
the ControlSignal by assigning ``None`` to its argument in the constructor (or the corresponding entry in its params
dictionary).  If a cost function is permanently disabled for a ControlSignal, it cannot be re-enabled using
``toggle_cost_function``.

In addition to its functions, a ControlSignal projection has an ``allocation_samples`` attribute.  This is a list
by :ref:`ControlMechanisms <ControlMechanism> that sample different values of ``allocation`` in order to adaptively
adjust the function of mechanisms in their systems (e.g., :doc:`EVCMechanism`).  The default value is an array that
ranges from 0.1 to 1 in steps of 0.1.

An attribute is assigned for each component of the cost (``intensityCost``, ``adjustmentCost``, and ``durationCost``),
the total cost (``cost``), and the resulting intensity (``intensity``).  In addition, the ``last_allocation`` and
``last_intensity`` attributes store the values associated with the previous execution of the projection.

.. _ControlSignal_Execution:

Execution
---------

A ControlSignal projection uses its ``function`` to compute its ``intensity``, and its :ref:`cost functions
<ControlSignal_Cost_Functions> use the ``intensity`` to compute the its ``cost``.  The ``intensity`` is assigned as
the ControlSignal projection's ``value``, which is used by the parmaterState to which it projects to modify the
corresponding parameter of the owner mechanism's function.

.. note::
   The changes in a parameter in response to the execution of a ControlSignal projection are not applied until the
   mechanism that receives the projection are next executed; see Lazy_Evaluation for an explanation of "lazy"
   updating).

.. _ControlSignal_Class_Reference:


Class Reference
---------------

"""

from PsyNeuLink.Components import DefaultController
# from Globals.Defaults import *
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Functions.Function import *

# # Default control allocation mode values:
# class DefaultControlAllocationMode(Enum):
#     GUMBY_MODE = 0.0
#     BADGER_MODE = 1.0
#     TEST_MODE = 240
# defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value
DEFAULT_ALLOCATION_SAMPLES = np.arange(0.1, 1.01, 0.1)

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

# ControlSignal Function Names
CONTROL_SIGNAL_COST_OPTIONS = 'controlSignalCostOptions'

INTENSITY_COST_FUNCTION = 'intensity_cost_function'
ADJUSTMENT_COST_FUNCTION = 'adjustment_cost_function'
DURATION_COST_FUNCTION = 'duration_cost_function'
COST_COMBINATION_FUNCTION = 'cost_combination_function'
costFunctionNames = [INTENSITY_COST_FUNCTION,
                     ADJUSTMENT_COST_FUNCTION,
                     DURATION_COST_FUNCTION,
                     COST_COMBINATION_FUNCTION]

# Attributes / KVO keypaths
# kpLog = "Control Signal Log"
kpAllocation = "Control Signal Allocation"
kpIntensity = "Control Signal Intensity"
kpCostRange = "Control Signal Cost Range"
kpIntensityCost = "Control Signal Intensity Cost"
kpAdjustmentCost = "Control Signal Adjustment Cost"
kpDurationCost = "Control Signal DurationCost"
kpCost = "Control Signal Cost"

class ControlSignalCostOptions(IntEnum):
    NONE               = 0
    INTENSITY_COST     = 1 << 1
    ADJUSTMENT_COST    = 1 << 2
    DURATION_COST      = 1 << 3
    ALL                = INTENSITY_COST | ADJUSTMENT_COST | DURATION_COST
    DEFAULTS           = INTENSITY_COST

ControlSignalValuesTuple = namedtuple('ControlSignalValuesTuple','intensity cost')

ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)



# IMPLEMENTATION NOTE:  ADD DESCRIPTION OF ControlSignal CHANNELS:  ADDED TO ANY SENDER OF A ControlSignal Projection:
    # USED, AT A MININUM, FOR ALIGNING VALIDATION OF inputStates WITH ITEMS IN variable
    #                      ?? AND SAME FOR FOR outputStates WITH value
    # SHOULD BE INCLUDED IN INSTANTIATION OF CONTROL MECHANISM (per SYSTEM DEFAULT CONTROL MECHANISM)
    #     IN OVERRIDES OF _validate_variable AND
    #     ?? WHEREVER variable OF outputState IS VALIDATED AGAINST value (search for FIX)

# class ControlSignal_Base(Projection_Base):
class ControlSignal(Projection_Base):
    """
    ControlSignal(                                   \
     sender=None,                                    \
     receiver=None,                                  \
     function=Linear                                 \
     intensity_cost_function=Exponential,            \
     adjustment_cost_function=Linear,                \
     duration_cost_function=Integrator,              \
     cost_combination_function=LinearCombination,    \
     allocation_samples=DEFAULT_ALLOCATION_SAMPLES,  \
     params=None,                                    \
     name=None,                                      \
     prefs=None)

    Implements a projection that controls the parameter of a mechanism function.

    COMMENT:
        Description:
            The ControlSignal class is a type in the Projection category of Component,
            It:
               - takes an allocation (scalar) as its input (self.variable)
               - uses self.function (params[FUNCTION]) to compute intensity based on allocation from self.sender,
                   used by self.receiver.owner to modify a parameter of self.receiver.owner.function

        ** MOVE:
        ProjectionRegistry:
            All ControlSignal projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

        Class attributes:
            + color (value): for use in interface design
            + classPreference (PreferenceSet): ControlSignalPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + paramClassDefaults:
                FUNCTION:Linear,
                FUNCTION_PARAMS:{SLOPE: 1, INTERCEPT: 0},  # Note: this implements identity function
                PROJECTION_SENDER: DefaultController, # ControlSignal (assigned to class ref in __init__ module)
                PROJECTION_SENDER_VALUE: [defaultControlAllocation],
                CONTROL_SIGNAL_COST_OPTIONS:ControlSignalCostOptions.DEFAULTS,
                kwControlSignalLogProfile: ControlSignalLog.DEFAULTS,
                ALLOCATION_SAMPLES: DEFAULT_ALLOCATION_SAMPLES,
            + paramNames = paramClassDefaults.keys()
            + costFunctionNames = paramClassDefaults[kwControlSignalCostFunctions].keys()
    COMMENT

    Arguments
    ---------

    sender : Optional[Mechanism or OutputState]
        Source of the allocation for the ControlSignal;  usually an outputState of a :doc:`ControlMechanism`.
        If it is not specified, the :doc:`DefaultControlMechanism` for the system to which the receiver belongs
        will be assigned.

    receiver : Optional[Mechanism or ParameterState]
        The parameterState associated with the parameter of a function to be controlled.  This must be specified,
        or be able to be determined by the context in which the ControlSignal is created or assigned.

    function : TransferFunction : default Linear
        Converts the ControlSignal's ``allocation`` into its ``intensity`` (equal to its ``value``).

    intensity_cost_function : Optional[TransferFuntion] : default Exponential
        Calculates a cost based on the ControlSignal's ``intensity``.
        It can be disabled permanently for the ControlSignal by assigning ``None``.

    adjustment_cost_function : Optiona[TransferFunction] : default Linear
        Calculates a cost based on a change in the ControlSignal's ``intensity`` from its last value.
        It can be disabled permanently for the ControlSignal by assigning ``None``.

    duration_cost_function : Optional[TransferFunction] : default Linear
        Calculates an integral of the ControlSignal's ``cost``.
        It can be disabled permanently for the ControlSignal by assigning ``None``.

    cost_combination_function : CombinationFunction : default LinearCombination
        Combines the results of any cost functions that are enabled, and assigns the result to ``cost``.

    allocation_samples : list : default :keyword:`DEFAULT_ALLOCATION_SAMPLES`
        Set of values used by ControlMechanisms that sample different allocation values in order to adaptively adjust
        the function of mechanisms in their systems.  The default value is an array that ranges from 0.1 to 1 in steps
        of 0.1.

    params : Optional[Dict[param keyword, param value]]
        Dictionary that can be used to specify the parameters for the projection, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).[LINK]
        By default, it contains an entry for the projection's default ``function`` and cost function assignments.

    name : str : default Transfer-<index>
        String used for the name of the ControlSignal projection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Process.classPreferences]
        Preference set for the ControlSignal projection.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    Attributes
    ----------

    COMMENT:
      ControlSignal_General_Attributes
    COMMENT

    allocation : float : default: defaultControlAllocation
        value used as ``variable`` for projection's ``function`` to determine ``intensity``.

    allocationSamples : list : DEFAULT_SAMPLE_VALUES
        set of values used by ControlMechanisms that sample different allocation values in order to
        adaptively adjust the function of mechanisms in their systems.

        .. _ControlSignal_Function_Attributes:

    function : TransferFunction :  default Linear
        converts ``allocation`` into ``intensity`` that is provided as output to receiver of projection.

    intensityCostFunction : TransferFunction : default Exponential
        calculates "intensityCost`` from the curent value of ``intensity``.

    adjustmentCostFunction : TransferFunction : default Linear
        calculates ``adjustmentCost`` based on the change in ``intensity`` from its last value.

    durationCostFunction : IntegratorFunction : default Linear
        calculates an integral of the ControlSignal's ``cost``.

    costCombinationFunction : CombinationFunction : default LinearCombination
        combines the results of any cost functions that are enabled, and assigns the result to ``cost``.

    COMMENT:
        ControlSignal_State_Attributes:
    COMMENT

    value : float
        during initialization, assigned keyword string (either INITIALIZING or DEFERRED_INITIALIZATION);
        during execution, returns value of ``intensity``.

    intensity : float
        output of ``function``, used to determine controlled parameter of task.

    intensityCost : float
        cost associated with current ``intensity``.

    adjustmentCost : float
        cost associated with last change to ``intensity``.

    durationCost
        intregral of ``cost``.

    cost : float
        current value of ControlSignal's ``cost``;  combined result of all cost functions that are enabled.

    COMMENT:
        ControlSignal_History_Attributes:
    COMMENT

    last_allocation : float
        ``allocation`` for last execution of the ControlSignal.

    last_intensity : float
        ``intensity`` for last execution of the ControlSignal.

        .. _ControlSignal_Cost_Functions:

    """

    color = 0

    componentType = CONTROL_SIGNAL
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    variableClassDefault = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_SENDER: DefaultController,
        PROJECTION_SENDER_VALUE: [defaultControlAllocation],
        CONTROL_SIGNAL_COST_OPTIONS:ControlSignalCostOptions.DEFAULTS})

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=Linear(slope=1, intercept=0),
                 intensity_cost_function:(is_function_type)=Exponential,
                 adjustment_cost_function:tc.optional(is_function_type)=Linear,
                 duration_cost_function:tc.optional(is_function_type)=Integrator,
                 cost_combination_function:tc.optional(is_function_type)=LinearCombination,
                 allocation_samples=DEFAULT_ALLOCATION_SAMPLES,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  intensity_cost_function=intensity_cost_function,
                                                  adjustment_cost_function=adjustment_cost_function,
                                                  duration_cost_function=duration_cost_function,
                                                  cost_combination_function=cost_combination_function,
                                                  allocation_samples=allocation_samples)

        # If receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if not receiver:
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name
            # Delete these as they have been moved to params dict (and so will not be recognized by Projection.__init__)
            del self.init_args[ALLOCATION_SAMPLES]
            del self.init_args[INTENSITY_COST_FUNCTION]
            del self.init_args[ADJUSTMENT_COST_FUNCTION]
            del self.init_args[DURATION_COST_FUNCTION]
            del self.init_args[COST_COMBINATION_FUNCTION]

            # Flag for deferred initialization
            self.value = DEFERRED_INITIALIZATION
            return

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        # super(ControlSignal_Base, self).__init__(sender=sender,
        super(ControlSignal, self).__init__(sender=sender,
                                            receiver=receiver,
                                            params=params,
                                            name=name,
                                            prefs=prefs,
                                            context=self)

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """validate allocation_samples and controlSignal cost functions

        Checks if:
        - cost functions are all appropriate
        - allocation_samples is a list with 2 numbers
        - all cost functions are references to valid ControlSignal costFunctions (listed in self.costFunctions)
        - IntensityFunction is identity function, in which case ignoreIntensityFunction flag is set (for efficiency)

        :param request_set:
        :param target_set:
        :param context:
        :return:
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
            #     others must be TransferFunction
            if isinstance(cost_function, Function):
                if cost_function_name == COST_COMBINATION_FUNCTION:
                    if not isinstance(cost_function, CombinationFunction):
                        raise ControlSignalError("Assignment of Function to {} ({}) must be a CombinationFunction".
                                                 format(COST_COMBINATION_FUNCTION, cost_function))
                elif not isinstance(cost_function, TransferFunction):
                    raise ControlSignalError("Assignment of Function to {} ({}) must be a TransferFunction".
                                             format(cost_function_name, cost_function))

            # cost_function is custom-specified function
            #     COST_COMBINATION_FUNCTION must accept an array
            #     others must accept a scalar
            #     all must return a scalar
            elif isinstance(cost_function, function_type):
                if cost_function_name == COST_COMBINATION_FUNCTION:
                    test_value = [1, 1]
                else:
                    test_value = 1
                try:
                    if not is_numerical(cost_function()):
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
            raise ControlSignalError("allocation_samples argument ({}) in {} must be a list or 1D np.array of number".
                                     format(allocation_samples, self.name))


        super()._validate_params(request_set=request_set,
                                                   target_set=target_set,
                                                   context=context)

        # ControlSignal Cost Functions
        for cost_function_name in costFunctionNames:
            cost_function = target_set[cost_function_name]
            if not cost_function:
                continue
            if not isinstance(cost_function, Function) and not issubclass(cost_function, Function):
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
        self.last_allocation = self.allocation
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

        self.intensity = self.function(self.allocation)
        self.last_intensity = self.intensity

    def _instantiate_sender(self, context=None):
# FIX: NEEDS TO BE BETTER INTEGRATED WITH super()._instantiate_sender
        """Check if DefaultController is being assigned and if so configures it for the requested ControlSignal

        If self.sender is a Mechanism, re-assign to <Mechanism>.outputState
        Insure that sender.value = self.variable

        This method overrides the corresponding method of Projection, before calling it, to check if the
            DefaultController is being assigned as sender and, if so:
            - creates projection-dedicated inputState, outputState and ControlSignalChannel in DefaultController
            - puts them in DefaultController's inputStates, outputStates, and ControlSignalChannels attributes
            - lengthens variable of DefaultController to accommodate the ControlSignal channel
            - updates value of DefaultController (in resposne to new variable)
        Note: the default execute method of DefaultController simply maps the inputState value to the outputState

        :return:
        """

        if isinstance(self.sender, Process):
            raise ProjectionError("Illegal attempt to add a ControlSignal projection from a Process {0} "
                                  "to a mechanism {0} in pathway list".format(self.name, self.sender.name))

        # If sender is a class:
        # - assume it is Mechanism or State class ref (as validated in _validate_params)
        # - implement default sender of the corresponding type
        if inspect.isclass(self.sender):
            # self.sender = self.paramsCurrent[PROJECTION_SENDER](self.paramsCurrent[PROJECTION_SENDER_VALUE])
# FIX 6/28/16:  IF CLASS IS ControlMechanism SHOULD ONLY IMPLEMENT ONCE;  THEREAFTER, SHOULD USE EXISTING ONE
            self.sender = self.sender(self.paramsCurrent[PROJECTION_SENDER_VALUE])

# FIX:  THE FOLLOWING CAN BE CONDENSED:
# FIX:      ONLY TEST FOR ControlMechanism_Base (TO IMPLEMENT PROJECTION)
# FIX:      INSTANTATION OF OutputState WILL BE HANDLED IN CALL TO super._instantiate_sender
# FIX:      (CHECK TO BE SURE THAT THIS DOES NOT MUCK UP _instantiate_control_signal_projection FOR ControlMechanism)
        # If sender is a Mechanism (rather than a State) object, get (or instantiate) its State
        #    (Note:  this includes ControlMechanism)
        if isinstance(self.sender, Mechanism):
            # If sender is a ControlMechanism, call it to instantiate its controlSignal projection
            from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
            if isinstance(self.sender, ControlMechanism_Base):
                self.sender._instantiate_control_signal_projection(self, context=context)
        # Call super to instantiate sender
        super(ControlSignal, self)._instantiate_sender(context=context)

    def _instantiate_receiver(self, context=None):
        # FIX: THIS NEEDS TO BE PUT BEFORE _instantate_function SINCE THAT USES self.receiver
        """Handle situation in which self.receiver was specified as a Mechanism (rather than State)

        Overrides Projection._instantiate_receiver, to require that if the receiver is specified as a Mechanism, then:
            the receiver Mechanism must have one and only one ParameterState;
            otherwise, passes control to Projection._instantiate_receiver for validation

        :return:
        """
        if isinstance(self.receiver, Mechanism):
            # If there is just one param of ParameterState type in the receiver Mechanism
            # then assign it as actual receiver (which must be a State);  otherwise, raise exception
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            if len(dict((param_name, state) for param_name, state in self.receiver.paramsCurrent.items()
                    if isinstance(state, ParameterState))) == 1:
                receiver_parameter_state = [state for state in dict.values()][0]
                # Reassign self.receiver to Mechanism's parameterState
                self.receiver = receiver_parameter_state
                # # Add self as projection to that parameterState
                # # IMPLEMENTATION NOTE:
                # #   THIS SHOULD REALLY BE HANDLED BY THE Mechanism.add_projection METHOD, AS IT IS FOR inputStates
                # # # MODIFIED 6/22/16 OLD:
                # # self.receiver.receivesFromProjections.append(self)
                # # MODIFIED 6/22/16 NEW:
                # self.receiver.add_projection(projection=self, state=receiver_parameter_state, context=context)
            else:
                raise ControlSignalError("Unable to assign ControlSignal projection ({0}) from {1} to {2}, "
                                         "as it has several parameterStates;  must specify one (or each) of them"
                                         " as receiver(s)".
                                         format(self.name, self.sender.owner, self.receiver.name))
        # else:
        super(ControlSignal, self)._instantiate_receiver(context=context)

    def _compute_cost(self, intensity_cost, adjustment_cost, cost_combination_function):
        """Compute the current cost for the control signal, based on allocation and most recent adjustment

        Computes the current cost by combining intensityCost and adjustmentCost, using the function specified by
              cost_combination_function (should be of Function type; default: LinearCombination)
        Returns totalCost

        :parameter intensity_cost
        :parameter adjustment_cost:
        :parameter cost_combination_function: (should be of Function type)
        :returns cost:
        :rtype: scalar:
        """

        return cost_combination_function([intensity_cost, adjustment_cost])

    def execute(self, variable=NotImplemented, params=NotImplemented, time_scale=None, context=None):
        """Adjust the control signal, based on the allocation value passed to it

        Computes new intensity and cost attributes from allocation
        Returns ControlSignalValuesTuple (intensity, totalCost)

        Use self.function to assign intensity
            - if ignoreIntensityFunction is set (for effiency, if the the execute method it is the identity function):
                ignore self.function
                pass allocation (input to controlSignal) along as its output
        Update cost

        :parameter allocation: (single item list, [0-1])
        :return: (intensity)
        """

        # store previous state
        self.last_allocation = self.allocation
        self.last_intensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.durationCost

        # update current intensity
        # FIX: IS THIS CORRECT?? OR SHOULD IT INDEED BE self.variable?
        # self.allocation = variable
        self.allocation = self.sender.value

        if self.ignoreIntensityFunction:
            # self.set_intensity(self.allocation)
            self.intensity = self.allocation
        else:
            self.intensity = self.function(self.allocation, params)
        intensity_change = self.intensity-self.last_intensity

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
        new_cost = 0
        if self.controlSignalCostOptions & ControlSignalCostOptions.INTENSITY_COST:
            new_cost = self.intensityCost = self.intensityCostFunction(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")
        if self.controlSignalCostOptions & ControlSignalCostOptions.ADJUSTMENT_COST:
            self.adjustmentCost = self.adjustmentCostFunction(intensity_change)
            new_cost = self._compute_cost(self.intensityCost,
                                         self.adjustmentCost,
                                         self.costCombinationFunction)
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")
        if self.controlSignalCostOptions & ControlSignalCostOptions.DURATION_COST:
            self.durationCost = self.durationCostFunction([self.last_duration_cost, new_cost])
            new_cost += self.durationCost
            if self.prefs.verbosePref:
                print("++ Used duration cost")
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

        #region Record controlSignal values in receiver mechanism log
        # Notes:
        # * Log controlSignals for ALL states of a given mechanism in the mechanism's log
        # * Log controlSignals for EACH state in a separate entry of the mechanism's log

        # Get receiver mechanism and state
        receiver_mech = self.receiver.owner
        receiver_state = self.receiver

        # Get logPref for mechanism
        log_pref = receiver_mech.prefs.logPref

        # Get context
        if not context:
            context = receiver_mech.name + " " + self.name + kwAssign
        else:
            context = context + kwSeparatorBar + self.name + kwAssign

        # If context is consistent with log_pref:
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and kwExecuting in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (kwExecuting in context))):
            # record info in log

# FIX: ENCODE ALL OF THIS AS 1D ARRAYS IN 2D PROJECTION VALUE, AND PASS TO .value FOR LOGGING
            receiver_mech.log.entries[receiver_state.name + " " +
                                      kpIntensity] = LogEntry(CurrentTime(), context, float(self.intensity))
            if not self.ignoreIntensityFunction:
                receiver_mech.log.entries[receiver_state.name + " " + kpAllocation] =     \
                    LogEntry(CurrentTime(), context, float(self.allocation))
                receiver_mech.log.entries[receiver_state.name + " " + kpIntensityCost] =  \
                    LogEntry(CurrentTime(), context, float(self.intensityCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpAdjustmentCost] = \
                    LogEntry(CurrentTime(), context, float(self.adjustmentCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpDurationCost] =   \
                    LogEntry(CurrentTime(), context, float(self.durationCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpCost] =           \
                    LogEntry(CurrentTime(), context, float(self.cost))
    #endregion

        return self.intensity

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
            raise ControlSignalError("AUTO not yet supported for {} param of ControlSignal; default will be used".
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

        ``cost_function_name`` should be a keyword (list under :ref:`Structure <ControlSignal_Structure>`).
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


# def RegisterControlSignal():
#     ProjectionRegistry(ControlSignal)
