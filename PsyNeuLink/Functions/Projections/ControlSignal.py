# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  ControlSignal *********************************************************
#

from PsyNeuLink.Functions import DefaultController
# from Globals.Defaults import *
from PsyNeuLink.Functions.Projections.Projection import *
from PsyNeuLink.Functions.Utilities.Utility import *

# # Default control allocation mode values:
# class DefaultControlAllocationMode(Enum):
#     GUMBY_MODE = 0.0
#     BADGER_MODE = 1.0
#     TEST_MODE = 240
# defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value

DEFAULT_ALLOCATION_SAMPLES = np.arange(0.1, 1.01, 0.1)

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

# Params:
kwControlSignalIdentity = "Control Signal Identity"
# kwControlSignalLogProfile = "Control Signal Log Profile"
kwControlSignalCostFunctions = "Control Signal Cost Functions"

# ControlSignal Function Names
kwControlSignalCosts = 'ControlSignalCosts'
kwControlSignalIntensityCostFunction = "Control Signal Intensity Cost Function"
kwControlSignalAdjustmentCostFunction = "Control Signal Adjustment Cost Function"
kwControlSignalDurationCostFunction = "Control Signal Duration Cost Function"
kwControlSignalTotalCostFunction = "Control Signal Total Cost Function"
kwControlSignalModulationFunction = "Control Signal Modulation Function"

# Attributes / KVO keypaths
# kpLog = "Control Signal Log"
kpAllocation = "Control Signal Allocation"
kpIntensity = "Control Signal Intensity"
kpCostRange = "Control Signal Cost Range"
kpIntensityCost = "Control Signal Intensity Cost"
kpAdjustmentCost = "Control Signal Adjustment Cost"
kpDurationCost = "Control Signal DurationCost"
kpCost = "Control Signal Cost"

class ControlSignalCosts(IntEnum):
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
    """Implement projection that controls a parameter value (default: IdentityMapping)

    Description:
        The ControlSignal class is a functionType in the Projection category of Function,
        It:
           - takes an allocation (scalar) as its input (self.variable)
           - uses self.function (params[FUNCTION]) to compute intensity based on allocation from self.sender,
               used by self.receiver.owner to modify a parameter of self.receiver.owner.function

    Instantiation:
        - ControlSignals can be instantiated in one of several ways:
            - directly: requires explicit specification of the receiver
            - as part of the instantiation of a mechanism:
                each parameter of a mechanism will, by default, instantiate a ControlSignal projection
                   to its State, using this as ControlSignal's receiver
            [TBI: - in all cases, the default sender of a Control is the EVC mechanism]

    Initialization arguments:
        - allocation (number) - source of allocation value (default: DEFAULT_ALLOCATION) [TBI: DefaultController]
        - receiver (State) - associated with parameter of mechanism to be modulated by ControlSignal
        - params (dict):
# IMPLEMENTATION NOTE: WHY ISN'T kwProjectionSenderValue HERE AS FOR Mapping??
            + FUNCTION (Utility): (default: Linear):
                determines how allocation (variable) is translated into the output
            + FUNCTION_PARAMS (dict): (default: {SLOPE: 1, INTERCEPT: 0}) - Note: implements identity function
            + kwControlSignalIdentity (list): vector that uniquely identifies the signal (default: NotImplemented)
            + kwAllocationSamples (list):
                list of allocation values to be sampled for ControlSignal (default: DEFAULT_ALLOCATION_SAMPLES)
            + kwControlSignalCostFunctions (dict): (default: NotImplemented - uses refs in paramClassDefaults)
                determine how costs are computed
                the key for each entry must be the name of a control signal cost function (see below) and
                the value must be a function initialization call (with optional variable and params dict args)
                Format: {<kwControlSignalCostFunctionName:<functionName(variable, params, <other args>)}
                    + kwControlSignalIntensityCostFunction: (default: Exponential)
                    + kwControlSignalAdjustmentCostFunction: (default: Linear) 
                    + kwControlSignalDurationCostFunction: (default: Linear)  
                    + kwControlSignalTotalCostFunction: (default: LinearCombination)

# IMPLEMENTATION NOTE:  ?? IS THIS STILL CORRECT?  IF NOT, SEARCH FOR AND CORRECT IN OTHER CLASSES
        # - name (str) - must be name of subclass;  otherwise raises an exception for direct call
        - name (str) - name of control signal (default: kwControlSignalDefaultName)
        - [TBI: prefs (dict)]
        # - logProfile (LogProfile enum): controls logging behavior (default: LogProfile.DEFAULTS)
        - context (str) - optional (default: NotImplemented)

    ProjectionRegistry:
        All ControlSignal projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        ControlSignal projections can be named explicitly (using the name='<name>' argument).  If this argument
           is omitted, it will be assigned "ControlSignal" with a hyphenated, indexed suffix ('ControlSignal-n')

    Class attributes:
        + color (value): for use in interface design
        + classPreference (PreferenceSet): ControlSignalPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + paramClassDefaults:
            FUNCTION:Linear,
            FUNCTION_PARAMS:{SLOPE: 1, INTERCEPT: 0},  # Note: this implements identity function
            kwProjectionSender: DefaultController, # ControlSignal (assigned to class ref in __init__ module)
            kwProjectionSenderValue: [defaultControlAllocation],
            kwControlSignalIdentity: NotImplemented,
            kwControlSignalCosts:ControlSignalCosts.DEFAULTS,
            kwControlSignalLogProfile: ControlSignalLog.DEFAULTS,
            kwAllocationSamples: DEFAULT_ALLOCATION_SAMPLES,
            kwControlSignalCostFunctions: {
                           kwControlSignalIntensityCostFunction: Exponential(context="ControlSignalIntensityCostFunction"),
                           kwControlSignalAdjustmentCostFunction: Linear(context="ControlSignalAjdustmentCostFunction"),
                           kwControlSignalDurationCostFunction: Linear(context="ControlSignalDurationCostFunction"),
                           kwControlSignalTotalCostFunction: LinearCombination(context="ControlSignalTotalCostFunction")
                                       }})
        + paramNames = paramClassDefaults.keys()
        + costFunctionNames = paramClassDefaults[kwControlSignalCostFunctions].keys()


    Instance attributes:
        General attributes
        + variable (value) - used as input to projection's execute method
        + allocationSamples - either the keyword AUTO (the default; samples are computed automatically);
                            a list specifying the samples to be evaluated;
                            or DEFAULT or NotImplemented (in which it uses a list
                            generated from DEFAULT_SAMPLE_VALUES)
        State attributes:
            - intensity -- value used to determine controlled parameter of task
            - intensityCost -- cost associated with current intensity
            - adjustmentCost -- cost associated with last change to intensity
            - durationCost - cost associated with temporal integral of intensity
            - cost -- curent value of total cost
        History attributes -- used to compute costs of changes to control signal:
            + last_allocation
            + last_intensity
        Cost Functions -- used to compute cost:
            + FUNCTION - converts allocation into intensity that is provided as output to receiver of projection
            + IntensityCostFunction -- converts intensity into its contribution to the cost
            + AdjustmentCostFunction -- converts change in intensity into its contribution to the cost
            + DurationCostFunction -- converts duration of control signal into its contribution to the cost
            + TotalCostFunction -- combines intensity and adjustment costs into reported cost
            NOTE:  there are class variables for each type of function that list the functions allowable for each type

        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying ControlSignalPreferenceSet

    Instance methods:
        - update_control_signal(allocation) -- computes new intensity and cost attributes from allocation
                                          - returns ControlSignalValuesTuple (intensity, totalCost)
        - compute_cost(self, intensity_cost, adjustment_cost, total_cost_function)
            - computes the current cost by combining intensityCost and adjustmentCost, using function specified by
              total_cost_function (should be of Function type; default: LinearCombination)
            - returns totalCost
        - log_all_entries - logs the entries specified in the log_profile attribute
        - assign_function(self, control_function_type, function_name, variables params)
            - (re-)assigns a specified function, including an optional parameter list
        - set_log - enables/disables automated logging
        - set_log_profile - assigns settings specified in the logProfile param (an instance of LogProfile)
        - set_allocation_samples
        - get_ignoreIntensityFunction
        - set_intensity_cost - enables/disables use of the intensity cost
        - get_intensity_cost
        - set_adjustment_cost - enables/disables use of the adjustment cost
        - get_adjust
        - set_duration_cost - enables/disables use of the duration cost
        - get_duration_cost
        - get_costs - returns three-element list with intensityCost, adjustmentCost and durationCost
    """

    color = 0

    functionType = CONTROL_SIGNAL
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    variableClassDefault = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        # FUNCTION_PARAMS:{PARAMETER_STATES: None}, # This suppresses parameterStates
        kwProjectionSender: DefaultController,
        kwProjectionSenderValue: [defaultControlAllocation],
        kwControlSignalIdentity: NotImplemented,
        kwControlSignalCosts:ControlSignalCosts.DEFAULTS,
        kwControlSignalCostFunctions: {
                       kwControlSignalIntensityCostFunction: Exponential(context="ControlSignalIntensityCostFunction"),
                       kwControlSignalAdjustmentCostFunction: Linear(context="ControlSignalAjdustmentCostFunction"),
                       kwControlSignalDurationCostFunction: Linear(context="ControlSignalDurationCostFunction"),
                       kwControlSignalTotalCostFunction: LinearCombination(context="ControlSignalTotalCostFunction")
                                   }})
    costFunctionNames = paramClassDefaults[kwControlSignalCostFunctions].keys()

    @tc.typecheck
    def __init__(self,
                 sender=NotImplemented,
                 receiver=NotImplemented,
                 function=Linear(slope=1, intercept=0),
                 allocation_samples=DEFAULT_ALLOCATION_SAMPLES,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """

        :param sender: (list)
        :param receiver: (list)
        :param params: (dict)
        :param name: (str)
        :param prefs: (dict)
        :param context: (str)
        :return:
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                 allocation_samples=allocation_samples)

        # If receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if receiver is NotImplemented:
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name
            # Delete this as it has been moved to params dict (and will not be recognized by Projection.__init__)
            del self.init_args[kwAllocationSamples]

            # Flag for deferred initialization
            self.value = kwDeferredInit
            return

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of functionName in super.__init__)
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
        - allocation_samples is a list with 2 numbers
        - all cost functions are references to valid ControlSignal costFunctions (listed in self.costFunctions)
        - IntensityFunction is identity function, in which case ignoreIntensityFunction flag is set (for efficiency)

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        # Validate allocation samples list:
        # - default is 1D np.array (defined by DEFAULT_ALLOCATION_SAMPLES)
        # - however, for convenience and compatibility, allow lists:
        #    check if it is a list of numbers, and if so convert to np.array
        allocation_samples = request_set[kwAllocationSamples]
        if isinstance(allocation_samples, list):
            if iscompatible(allocation_samples, **{kwCompatibilityType: list,
                                                       kwCompatibilityNumeric: True,
                                                       kwCompatibilityLength: False,
                                                       }):
                # Convert to np.array to be compatible with default value
                request_set[kwAllocationSamples] = np.array(allocation_samples)
        elif isinstance(allocation_samples, np.ndarray) and allocation_samples.ndim == 1:
            pass
        else:
            raise ControlSignalError("allocation_samples argument ({}) in {} must be a list or 1D np.array of number".
                                     format(allocation_samples, self.name))


        super()._validate_params(request_set=request_set,
                                                   target_set=target_set,
                                                   context=context)

        # ControlSignal Cost Functions
        if target_set[kwControlSignalCostFunctions]:
            for function_name, function in request_set[kwControlSignalCostFunctions].items():
                # self.assign_function(function_name,function)
                if not issubclass(type(function), Function):
                    raise ControlSignalError("{0} not a valid Function".format(function))

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        for function in self.paramsCurrent[kwControlSignalCostFunctions].values():
            function.owner = self

        self.controlSignalCosts = self.paramsCurrent[kwControlSignalCosts]

        # Assign instance attributes
        self.controlIdentity = self.paramsCurrent[kwControlSignalIdentity]
        self.allocationSamples = self.paramsCurrent[kwAllocationSamples]
        self.costFunctions = self.paramsCurrent[kwControlSignalCostFunctions]

        # Default intensity params
        self.default_allocation = defaultControlAllocation
        self.allocation = self.default_allocation  # Amount of control currently licensed to this signal
        self.last_allocation = self.allocation
        self.intensity = self.allocation

        # Default cost params
        self.intensityCost = self.costFunctions[kwControlSignalIntensityCostFunction].function(self.intensity)
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

    def instantiate_sender(self, context=None):
# FIX: NEEDS TO BE BETTER INTEGRATED WITH super().instantiate_sender
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
            # self.sender = self.paramsCurrent[kwProjectionSender](self.paramsCurrent[kwProjectionSenderValue])
# FIX 6/28/16:  IF CLASS IS ControlMechanism SHOULD ONLY IMPLEMENT ONCE;  THEREAFTER, SHOULD USE EXISTING ONE
            self.sender = self.sender(self.paramsCurrent[kwProjectionSenderValue])

# FIX:  THE FOLLOWING CAN BE CONDENSED:
# FIX:      ONLY TEST FOR ControlMechanism_Base (TO IMPLEMENT PROJECTION)
# FIX:      INSTANTATION OF OutputState WILL BE HANDLED IN CALL TO super.instantiate_sender
# FIX:      (CHECK TO BE SURE THAT THIS DOES NOT MUCK UP instantiate_control_signal_projection FOR ControlMechanism)
        # If sender is a Mechanism (rather than a State) object, get (or instantiate) its State
        #    (Note:  this includes ControlMechanism)
        if isinstance(self.sender, Mechanism):
            # If sender is a ControlMechanism, call it to instantiate its controlSignal projection
            from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
            if isinstance(self.sender, ControlMechanism_Base):
                self.sender.instantiate_control_signal_projection(self, context=context)
        # Call super to instantiate sender
        super(ControlSignal, self).instantiate_sender(context=context)

    def instantiate_receiver(self, context=None):
        """Handle situation in which self.receiver was specified as a Mechanism (rather than State)

        Overrides Projection.instantiate_receiver, to require that if the receiver is specified as a Mechanism, then:
            the receiver Mechanism must have one and only one ParameterState;
            otherwise, passes control to Projection.instantiate_receiver for validation

        :return:
        """
        if isinstance(self.receiver, Mechanism):
            # If there is just one param of ParameterState type in the receiver Mechanism
            # then assign it as actual receiver (which must be a State);  otherwise, raise exception
            from PsyNeuLink.Functions.States.ParameterState import ParameterState
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
        super(ControlSignal, self).instantiate_receiver(context=context)

    def compute_cost(self, intensity_cost, adjustment_cost, total_cost_function):
        """Compute the current cost for the control signal, based on allocation and most recent adjustment

            :parameter intensity_cost
            :parameter adjustment_cost:
            :parameter total_cost_function: (should be of Function type)
            :returns cost:
            :rtype: scalar:
        """

        return total_cost_function.function([intensity_cost, adjustment_cost])

    def execute(self, variable=NotImplemented, params=NotImplemented, time_scale=None, context=None):
        """Adjust the control signal, based on the allocation value passed to it

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
        if self.controlSignalCosts & ControlSignalCosts.INTENSITY_COST:
            new_cost = self.intensityCost = self.costFunctions[kwControlSignalIntensityCostFunction].function(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")
        if self.controlSignalCosts & ControlSignalCosts.ADJUSTMENT_COST:
            self.adjustmentCost = self.costFunctions[kwControlSignalAdjustmentCostFunction].function(intensity_change)
            new_cost = self.compute_cost(self.intensityCost,
                                         self.adjustmentCost,
                                         self.costFunctions[kwControlSignalTotalCostFunction])
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")
        if self.controlSignalCosts & ControlSignalCosts.DURATION_COST:
            self.durationCost = self.costFunctions[kwControlSignalDurationCostFunction].function([self.last_duration_cost,
                                                                                             new_cost])
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

# IMPLEMENTATION NOTE:  *** SHOULDN'T THIS JUST USE ASSIGN_DEFAULT (OR ADJUST) PROPERTY OF FUNCTION?? x

    def assign_function(self, control_signal_function_name, function):
        # ADD DESCDRIPTION HERE:  NOTE THAT function_type MUST BE A REFERENCE TO AN INSTANCE OF A FUNCTION

        if not issubclass(type(function), Function):
            raise ControlSignalError("{0} not a valid Function".format(function))
        else:
            self.paramsCurrent[kwControlSignalCostFunctions][control_signal_function_name] = function
            self.costFunctions[control_signal_function_name] = function

# Fix: rewrite this all with @property
    # Setters and getters

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
            raise ControlSignalError("AUTO not yet support for {} param of ControlSignal; default will be used".
                                     format(kwAllocationSamples))
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

    def set_intensity_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.INTENSITY_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.INTENSITY_COST

    def set_adjustment_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.ADJUSTMENT_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.ADJUSTMENT_COST

    def set_duration_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.DURATION_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.DURATION_COST

    def get_costs(self):
        return [self.intensityCost, self.adjustmentCost, self.durationCost]


# def RegisterControlSignal():
#     ProjectionRegistry(ControlSignal)
