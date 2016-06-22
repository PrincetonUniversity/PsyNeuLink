#
# *********************************************  ControlSignal *********************************************************
#

from Functions import SystemDefaultController
# from Globals.Defaults import *
from Functions.Projections.Projection import *
from Functions.Utility import *

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

# Params:

kwControlSignalIdentity = "Control Signal Identity"
kwControlSignalLogProfile = "Control Signal Log Profile"
kwControlSignalAllocationSamplingRange = "Control Signal Allocation Sampling Range"
kwControlSignalFunctions = "Control Signal Functions"

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
    DEFAULTS           = NONE

ControlSignalValuesTuple = namedtuple('ControlSignalValuesTuple','intensity cost')

ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


# class LogEntry(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 1
#     ALLOCATION      = 1 << 1
#     INTENSITY       = 1 << 2
#     INTENSITY_COST  = 1 << 3
#     ADJUSTMENT_COST = 1 << 4
#     DURATION_COST   = 1 << 5
#     COST            = 1 << 6
#     ALL_COSTS = INTENSITY_COST | ADJUSTMENT_COST | DURATION_COST | COST
#     ALL = TIME_STAMP | ALLOCATION | INTENSITY | ALL_COSTS
#     DEFAULTS = NONE


# class ControlSignalLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALLOCATION      = 1 << 1
#     INTENSITY       = 1 << 2
#     INTENSITY_COST  = 1 << 3
#     ADJUSTMENT_COST = 1 << 4
#     DURATION_COST   = 1 << 5
#     COST            = 1 << 6
#     ALL_COSTS = INTENSITY_COST | ADJUSTMENT_COST | DURATION_COST | COST
#     ALL = TIME_STAMP | ALLOCATION | INTENSITY | ALL_COSTS
#     DEFAULTS = NONE
#

class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)



# IMPLEMENTATION NOTE:  ADD DESCRIPTION OF ControlSignal CHANNELS:  ADDED TO ANY SENDER OF A ControlSignal Projection:
    # USED, AT A MININUM, FOR ALIGNING VALIDATION OF inputStates WITH ITEMS IN variable
    #                      ?? AND SAME FOR FOR outputStates WITH executeMethodOutputDefault
    # SHOULD BE INCLUDED IN INSTANTIATION OF CONTROL MECHANISM (per SYSTEM DEFAULT CONTROL MECHANISM)
    #     IN OVERRIDES OF validate_variable AND
    #     ?? WHEREVER variable OF outputState IS VALIDATED AGAINST executeMethodOutputDefaut (search for FIX)

# class ControlSignal_Base(Projection_Base):
class ControlSignal(Projection_Base):
    """Implement projection that controls a parameter value (default: IdentityMapping)

    Description:
        The ControlSignal class is a functionType in the Projection category of Function,
        It:
           - takes an allocation (scalar) as its input (self.variable)
           - uses self.execute (params[kwExecuteMethod]) to compute an output from an allocation from self.sender,
               used by self.receiver.ownerMechanism to modify a parameter of self.receiver.ownerMechanism.function
        It must have all the attributes of a Projection object

    Instantiation:
        - ControlSignals can be instantiated in one of several ways:
            - directly: requires explicit specification of the receiver
            - as part of the instantiation of a mechanism:
                each parameter of a mechanism will, by default, instantiate a ControlSignal projection
                   to its MechanismState, using this as ControlSignal's receiver
            [TBI: - in all cases, the default sender of a Control is the EVC mechanism]

    Initialization arguments:
        - allocation (number) - source of allocation value (default: DEFAULT_ALLOCATION) [TBI: SystemDefaultController]
        - receiver (MechanismState) - associated with parameter of mechanism to be modulated by ControlSignal
        - params (dict):
# IMPLEMENTATION NOTE: WHY ISN'T kwProjectionSenderValue HERE AS FOR Mapping??
            + kwExecuteMethod (Utility): (default: Linear):
                determines how allocation (variable) is translated into the output
            + kwExecuteMethodParams (dict): (default: {kwSlope: 1, kwIntercept: 0}) - Note: implements identity function
            + kwControlSignalIdentity (list): vector that uniquely identifies the signal (default: NotImplemented)
            + kwControlSignalAllocationSamplingRange (2-item tuple):
                two element list that specifies search range for costs (default: NotImplemented)
            + kwControlSignalFunctions (dict): (default: NotImplemented - uses refs in paramClassDefaults)
                determine how allocation is converted to control signal intensity, and how costs are computed
                the key for each entry must be the name of a control signal function (see below) and
                the value must be a function initialization call (with optional variable and params dict args)
                Format: {<kwControlSignalFunctionName:<functionName(variable, params, <other args>)}
                    # + kwControlSignalIntensityFunction: (default: Linear, identity) - Replaced by kwExecuteMethod)
                    + kwControlSignalIntensityCostFunction: (default: Exponential) 
                    + kwControlSignalAdjustmentCostFunction: (default: Linear) 
                    + kwControlSignalDurationCostFunction: (default: Linear)  
                    + kwControlSignalTotalCostFunction: (default: Arithmetic) 

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
            kwExecuteMethod:Linear,
            kwExecuteMethodParams:{Linear.kwSlope: 1, Linear.kwIntercept: 0},  # Note: this implements identity function
            kwProjectionSender: SystemDefaultController, # ControlSignal (assigned to class ref in __init__ module)
            kwProjectionSenderValue: [defaultControlAllocation],
            kwControlSignalIdentity: NotImplemented,
            kwControlSignalCosts:ControlSignalCosts.DEFAULTS,
            kwControlSignalLogProfile: ControlSignalLog.DEFAULTS,
            kwControlSignalAllocationSamplingRange: NotImplemented,
            kwControlSignalFunctions: {
                           kwControlSignalIntensityCostFunction: Exponential(context="ControlSignalIntensityCostFunction"),
                           kwControlSignalAdjustmentCostFunction: Linear(context="ControlSignalAjdustmentCostFunction"),
                           kwControlSignalDurationCostFunction: Linear(context="ControlSignalDurationCostFunction"),
                           kwControlSignalTotalCostFunction: Arithmetic(context="ControlSignalTotalCostFunction")
                                       }})
        + paramNames = paramClassDefaults.keys()
        + functionNames = paramClassDefaults[kwControlSignalFunctions].keys()


    Instance attributes:
        General attributes
        + variable (value) - used as input to projection's execute method
        + allocationSamples - either the keyword AUTO (the default; samples are computed automatically);
                            a list specifying the samples to be evaluated;
                            or DEFAULT or NotImplemented (in which it uses a list
                            generated from DEFAULT_SAMPLE_VALUES)
        State attributes:
            - intensity — value used to determine controlled parameter of task
            - intensityCost — cost associated with current intensity
            - adjustmentCost — cost associated with last change to intensity
            - durationCost - cost associated with temporal integral of intensity
        History attributes — used to compute costs of changes to control signal:
            + last_allocation
            + last_intensity
        Functions — used to convert allocation into intensity, cost, and to modulate mechanism parameters:
            + kwExecuteMethod - converts allocation into intensity that is provided as output to receiver of projection
            # + IntensityFunction — converts allocation into intensity - Replaced by kwExecuteMethod
            + IntensityCostFunction — converts intensity into its contribution to the cost
            + AdjustmentCostFunction — converts change in intensity into its contribution to the cost
            + DurationCostFunction — converts duration of control signal into its contribution to the cost
            + TotalCostFunction — combines intensity and adjustment costs into reported cost
            # + ModulationFunction - determines how control influences mechanism parameter with which it is associated
            NOTE:  there are class variables for each type of function that list the functions allowable for each type

        + executeMethodOutputDefault (value) - sample output of projection's execute method
        + executeMethodOutputType (type) - type of output of projection's execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying ControlSignalPreferenceSet

    Instance methods:
            • update_control_signal(allocation) — computes new intensity and cost attributes from allocation
                                              - returns ControlSignalValuesTuple (intensity, totalCost)
            • compute_cost(self, intensity_cost, adjustment_cost, total_cost_function)
                - computes the current cost by combining intensityCost and adjustmentCost, using function specified by
                  total_cost_function (should be of Function type; default: Arithmetic)
                - returns totalCost
            • log_all_entries - logs the entries specified in the log_profile attribute
            • assign_function(self, control_function_type, function_name, variables params)
                - (re-)assigns a specified function, including an optional parameter list
            • set_log - enables/disables automated logging
            • set_log_profile - assigns settings specified in the logProfile param (an instance of LogProfile)
            • set_allocation_sampling_range
            • get_allocation_sampling_range
            • set_intensity
            • get_intensity
            • set_ignoreIntensityFunction - enables/disables use of intensity function (overrides automatic setting)
            • get_ignoreIntensityFunction
            • set_intensity_cost - enables/disables use of the intensity cost
            • get_intensity_cost
            • set_adjustment_cost - enables/disables use of the adjustment cost
            • get_adjust
            • set_duration_cost - enables/disables use of the duration cost
            • get_duration_cost
            • get_costs - returns three-element list with intensityCost, adjustmentCost and durationCost
    """

    color = 0

    functionType = kwControlSignal
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    variableClassDefault = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwExecuteMethod:Linear,
        kwExecuteMethodParams:{Linear.kwSlope: 1, Linear.kwIntercept: 0},  # Note: this implements identity function
        kwProjectionSender: SystemDefaultController, # Assigned to class ref in __init__ module
        kwProjectionSenderValue: [defaultControlAllocation],
        kwControlSignalIdentity: NotImplemented,
        kwControlSignalCosts:ControlSignalCosts.DEFAULTS,
        # kwControlSignalLogProfile: ControlSignalLog.DEFAULTS,
        kwControlSignalAllocationSamplingRange: NotImplemented,
        kwControlSignalFunctions: {
                       kwControlSignalIntensityCostFunction: Exponential(context="ControlSignalIntensityCostFunction"),
                       kwControlSignalAdjustmentCostFunction: Linear(context="ControlSignalAjdustmentCostFunction"),
                       kwControlSignalDurationCostFunction: Linear(context="ControlSignalDurationCostFunction"),
                       kwControlSignalTotalCostFunction: Arithmetic(context="ControlSignalTotalCostFunction")
                                   }})
    functionNames = paramClassDefaults[kwControlSignalFunctions].keys()

    def __init__(self,
                 allocation_source=NotImplemented,
                 receiver=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """

        :param allocation_source: (list)
        :param receiver: (list)
        :param params: (dict)
        :param name: (str)
        :param prefs: (dict)
        :param context: (str)
        :return:
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of functionName in super.__init__)
        # super(ControlSignal_Base, self).__init__(sender=allocation_source,
        super(ControlSignal, self).__init__(sender=allocation_source,
                                            receiver=receiver,
                                            params=params,
                                            name=name,
                                            prefs=prefs,
                                            context=self)

        self.controlSignalCosts = self.paramsCurrent[kwControlSignalCosts]

        # Assign instance attributes
        self.controlIdentity = self.paramsCurrent[kwControlSignalIdentity]
        self.set_allocation_sampling_range(self.paramsCurrent[kwControlSignalAllocationSamplingRange])
        self.functions = self.paramsCurrent[kwControlSignalFunctions]

        # VALIDATE LOG PROFILE:
        # self.set_log_profile(self.paramsCurrent[kwControlSignalLogProfile])

        # KVO observer dictionary
        self.observers = {
            # kpLog: [],
            kpAllocation: [],
            kpIntensity: [],
            kpIntensityCost: [],
            kpAdjustmentCost: [],
            kpDurationCost: [],
            kpCost: []
        }

        # Default intensity params
        self.default_allocation = defaultControlAllocation
        self.allocation = self.default_allocation  # Amount of control currently licensed to this signal
        self.last_allocation = self.allocation
        self.intensity = 0 # Needed to define attribute
        self.set_intensity(self.execute(self.allocation))
        self.last_intensity = self.intensity
        # if self.functions[kwControlSignalIntensityFunction].functionName == kwLinear and \
        #      self.functions[kwControlSignalIntensityFunction].paramsCurrent[Linear.kwSlope] == 1 and \
        #      self.functions[kwControlSignalIntensityFunction].paramsCurrent[Linear.kwIntercept] == 1 == 0:
        if (isinstance(self.execute, Linear) and
                    self.execute.paramsCurrent[Linear.kwSlope] is 1 and
                    self.execute.paramsCurrent[Linear.kwIntercept] is 0):
             self.ignoreIntensityFunction = True
        else:
            self.ignoreIntensityFunction = False
        # print("Ignore intensity function: ",self.ignoreIntensityFunction)
        # print("Function type name: ", self.functions[kwControlSignalIntensityFunction].functionType)
        # print("Function type name tested: ",Functions.kwLinear)
        # print("Slope: ", self.functions[kwControlSignalIntensityFunction].paramsCurrent[Linear.kwSlope])
        # print("intercept: ", self.functions[kwControlSignalIntensityFunction].paramsCurrent[Linear.kwIntercept])

        # Default cost params
        self.intensityCost = self.functions[kwControlSignalIntensityCostFunction].execute(self.intensity)
        self.adjustmentCost = 0
        self.durationCost = 0
        self.last_duration_cost = self.durationCost
        self.cost = self.intensityCost
        self.last_cost = self.cost

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """validate allocation_sampling_range and controlSignal functions

        Checks if:
        - allocation_sampling_range is a list with 2 numbers
        - all functions are references to valid ControlSignal functions (listed in self.functions)
        - IntensityFunction is identity function, in which case ignoreIntensityFunction flag is set (for efficiency)

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        super(ControlSignal, self).validate_params(request_set=request_set,
                                                   target_set=target_set,
                                                   context=context)

        # Allocation Sampling Range
        alloc_sample_range = target_set[kwControlSignalAllocationSamplingRange]

        if not iscompatible(alloc_sample_range, **{kwCompatibilityType: list,
                                                   kwCompatibilityNumeric: True,
                                                   kwCompatibilityLength: 2}):
            raise ControlSignalError("allocation_sampling_range argument in {0} must be a list with two numbers".
                                     format(self.name))

        # ControlSignal Functions
        if target_set[kwControlSignalFunctions]:
            for function_name, function in request_set[kwControlSignalFunctions].items():
                # self.assign_function(function_name,function)
                if not issubclass(type(function), Function):
                    raise ControlSignalError("Function type {0} not found in Functions.functionList".format(function))

        # If kwExecuteMethod (intensity function) is identity function, set ignoreIntensityFunction
        try:
            function = target_set[kwExecuteMethod]
        except KeyError:
            # IMPLEMENTATION NOTE:  put warning here that default ExecuteMethod will be used
            pass
        else:
            if (isinstance(function, Linear) and
                        function.paramsCurrent[Linear.kwSlope] == 1 and
                        function.paramsCurrent[Linear.kwIntercept] == 0):
                self.ignoreIntensityFunction = True
            else:
                self.ignoreIntensityFunction = False

    def instantiate_sender(self, context=NotImplemented):
        """Assign self to outputState of self.sender and insure compatibility with self.variable

        Assume self.sender has been assigned in validate_params, from either sender arg or kwProjectionSender

        If self.sender is a Mechanism, re-assign to <Mechanism>.outputState
        Insure that sender.executeMethodOutputDefault = self.variable

        This method overrides the corresponding method of Projection, before calling it, to check if the
            SystemDefaultController is being assigned as sender and, if so:
            - creates projection-dedicated inputState, outputState and ControlSignalChannel in SystemDefaultController
            - puts them in SystemDefaultController's inputStates, outputStates, and ControlSignalChannels attributes
            - lengthens variable of SystemDefaultController to accommodate the ControlSignal channel
            - updates executeMethodOutputDefault of SystemDefaultController (in resposne to new variable)
        Note: the default execute method of SystemDefaultController simply maps the inputState value to the outputState

        :return:
        """

        if isinstance(self.sender, Process):
            raise ProjectionError("Illegal attempt to add a ControlSignal projection from a Process {0} "
                                  "to a mechanism {0} in configuration list".format(self.name, self.sender.name))

        from collections import OrderedDict

        # If sender is a class:
        # - assume it is Mechanism or MechanismState (as validated in validate_params)
        # - implement default sender of the corresponding type
        if inspect.isclass(self.sender):
            # self.sender = self.paramsCurrent[kwProjectionSender](self.paramsCurrent[kwProjectionSenderValue])
            self.sender = self.sender(self.paramsCurrent[kwProjectionSenderValue])

        # If sender is a Mechanism (rather than a MechanismState) object, get (or instantiate) its MechanismState
        #    (Note:  this includes SystemDefaultController)
        if isinstance(self.sender, Mechanism):

            if kwSystemDefaultController in self.sender.name:
                self.sender.instantiate_control_signal_channels(self, context=context)

        # Call super to validate, set self.variable, and assign projection to sender's sendsToProjections atttribute
        super(ControlSignal, self).instantiate_sender(context=context)

    def instantiate_receiver(self, context=NotImplemented):
        """Assign ControlSignal projection to receiver

        Overrides Projection.instantiate_receiver, to require that if the receiver is specified as a Mechanism, then:
            the reciever Mechanism must have one and only one MechanismParameterState;
            otherwise, passes control to Projection.instantiate_receiver for validation

        :return:
        """
        if isinstance(self.receiver, Mechanism):
            # If there is just one param of MechanismParameterState type in the receiver
            # then assign it as receiver;  otherwise, raise exception
            from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
            if len(dict((param, value) for param, value in self.receiver.paramsCurrent.items()
                    if isinstance(value, MechanismParameterState))) == 1:
                self.receiver = [value for value in dict.values()][0]
                self.receiver.receivesFromProjections.append(self)

            else:
                raise ControlSignalError("Unable to assign ControlSignal projection ({0}) from {1} to {2}, "
                                         "as it has several parameterStates;  must specify one (or each) of them"
                                         " as receiver(s)".
                                         format(self.name, self.sender.ownerMechanism, self.receiver.name))
        else:
            super(ControlSignal, self).instantiate_receiver(context=context)

    def compute_cost(self, intensity_cost, adjustment_cost, total_cost_function):
        """Compute the current cost for the control signal, based on allocation and most recent adjustment

            :parameter intensity_cost
            :parameter adjustment_cost:
            :parameter total_cost_function: (should be of Function type)
            :returns cost:
            :rtype: scalar:
        """

        return total_cost_function.execute([intensity_cost, adjustment_cost])

    def update(self, params=NotImplemented, context=NotImplemented):
    # def update(self, params=NotImplemented, context=NotImplementedError):
        """Adjust the control signal, based on the allocation value passed to it

        :parameter allocation: (single item list, [0-1])
        # :return (intensity, cost):
        :return: (intensity)
        """

        allocation = self.sender.value

        # store previous state
        self.last_allocation = self.allocation
        self.last_intensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.durationCost

        # update current intensity
        self.allocation = allocation

        if self.ignoreIntensityFunction:
            self.set_intensity(self.allocation)
        else:
            # self.set_intensity(self.functions[kwControlSignalIntensityFunction].execution(allocation))
            self.set_intensity(self.execute(allocation, params))
        intensity_change = self.intensity-self.last_intensity
        if self.prefs.verbosePref:
            intensity_change_string = "no change"
            if intensity_change < 0:
                intensity_change_string = str(intensity_change)
            elif intensity_change > 0:
                intensity_change_string = "+" + str(intensity_change)
            if self.prefs.verbosePref:
                print("\nIntensity: {0} [{1}] (for allocation {2})".format(self.intensity,
                                                                                   intensity_change_string,
                                                                                   allocation))
                print("[Intensity function {0}]".format(["ignored", "used"][self.ignoreIntensityFunction]))

        # compute cost(s)
        new_cost = 0
        if self.controlSignalCosts & ControlSignalCosts.INTENSITY_COST:
            new_cost = self.intensityCost = self.functions[kwControlSignalIntensityCostFunction].execute(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")
        if self.controlSignalCosts & ControlSignalCosts.ADJUSTMENT_COST:
            self.adjustmentCost = self.functions[kwControlSignalAdjustmentCostFunction].execute(intensity_change)
            new_cost = self.compute_cost(self.intensityCost,
                                         self.adjustmentCost,
                                         self.functions[kwControlSignalTotalCostFunction])
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")
        if self.controlSignalCosts & ControlSignalCosts.DURATION_COST:
            self.durationCost = self.functions[kwControlSignalDurationCostFunction].execute([self.last_duration_cost,
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
        receiver_mech = self.receiver.ownerMechanism
        receiver_state = self.receiver

        # Get logPref for mechanism
        log_pref = receiver_mech.prefs.logPref

        # Get context
        if context is NotImplemented:
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
            raise ControlSignalError("Function type {0} not found in Functions.functionList".format(function))
        else:
            self.paramsCurrent[kwControlSignalFunctions][control_signal_function_name] = function
            self.functions[control_signal_function_name] = function

# Fix: rewrite this all with @property
    # Setters and getters

    def set_allocation_sampling_range(self, allocation_samples=DEFAULT):
        if isinstance(allocation_samples, list):
                self.allocationSamples = allocation_samples
        elif allocation_samples == AUTO:
            # THIS IS A STUB, TO BE REPLACED BY AN ACTUAL COMPUTATION OF THE ALLOCATION RANGE
            pass
        else:   # This is called if allocation_samples is 'DEFAULT' or not specified
            self.allocationSamples = []
            i = DEFAULT_ALLOCATION_SAMPLES[0]
            while i < DEFAULT_ALLOCATION_SAMPLES[1]:
                self.allocationSamples.append(i)
                i += DEFAULT_ALLOCATION_SAMPLES[2]

    def get_allocation_sampling_range(self):
        return self.allocationSamples

    def set_intensity(self, new_value):
        old_value = self.intensity
        self.intensity = new_value
        if len(self.observers[kpIntensity]):
            for observer in self.observers[kpIntensity]:
                observer.observe_value_at_keypath(kpIntensity, old_value, new_value)

    def get_intensity(self):
        return self.intensity

    def set_ignoreIntensityFunction(self, assignment=ON):
        if assignment:
            self.ignoreIntensityFunction = True
        else:
            self.ignoreIntensityFunction = False

    def get_ignoreIntensityFunction(self):
        return self.ignoreIntensityFunction

    def set_intensity_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.INTENSITY_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.INTENSITY_COST

    def get_intensity_cost(self):
        return self.intensityCost

    def set_adjustment_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.ADJUSTMENT_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.ADJUSTMENT_COST

    def get_adjustment_cost(self):
        return self.adjustmentCost

    def set_duration_cost(self, assignment=ON):
        if assignment:
            self.controlSignalCosts |= ControlSignalCosts.DURATION_COST
        else:
            self.controlSignalCosts &= ~ControlSignalCosts.DURATION_COST

    def get_duration_cost(self):
        return self.durationCost

    def get_costs(self):
        return [self.intensityCost, self.adjustmentCost, self.durationCost]


# def RegisterControlSignal():
#     ProjectionRegistry(ControlSignal)
