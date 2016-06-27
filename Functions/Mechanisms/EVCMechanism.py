#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanism_Base


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# NOTES: *******************************************************************************************************
#
# DOCUMENTATION:
#     kwMonitoredStates must be list of Mechanisms or MechanismOutputStates in Mechanisms that are in kwSystem
#     if Mechanism is specified in kwMonitoredStates, all of its outputStates are used
#     kwMonitoredStates assigns a Mapping Projection from each outputState to a newly created inputState in self.inputStates
#     executeMethod uses LinearCombination to apply a set of weights to the value of each monitored state to compute EVC
#     and then searches space of control signals (using allocation_sampling_range for each) to find combiantion that maxmizes EVC
#      OBJECTIVE FUNCTION FOR exeuteMethod:
#      Applies linear combination to values of monitored states (self.inputStates)
#      Execute method is LinearCombination, with weights = linear terms
#      kwExecuteMethodParams = kwWeights
#      Cost is summed over controlSignal costs and then uses kwCostFunction to combine value with cost
#
# IMPLEMENTATION:
#
# - IMPLEMENT: .add_projection(Mechanism or MechanismState) method
#                   that adds controlSignal projection from EVC to specified Mechanism/MechanismState
#                   validate that Mechanism / MechanismState.ownerMechanism is in self.system
#                   ? use Mechanism.add_projection method
# - IMPLEMENT: kwExecuteMethodParams for cost:  operation (additive or multiplicative), weight?
#
# - FIX: Make sure creation of controlSignalSearchSpace is correct (np.matrix)
#
# INSTANTIATION:
# - inputStates: one for each performance/environment variable monitored
# - specification of system:  required param: kwSystem
# - specification of inputStates:  required param: kwMonitoredStates
#
# - specification of executeMethod: default is default allocation policy (BADGER/GUMBY)
#     constraint:  if specified, number of items in variable must match number of inputStates in kwInputStates
#                  and names in list in kwMonitor must match those in kwInputStates
#
# EVALUATION:
# - evaluation function (as execute method) with one variable item (1D array) for each inputState
#      (??how should they be named/referenced:
#         maybe reverse instantation of variable and executeMethod, so that
#         execute method is parsed, and the necessary inputStates are created for it)
# - mapping projections from monitored states to inputStates
# - control signal projections established automatically by system implementation (using kwConrolSignal)
# - poll control signal projections for ranges to create matrix of search space
#
# EXECUTION:
# - call system.execute for each point in search space
# - compute evaluation function, and keep track of performance outcomes


class EVCMechanism(SystemControlMechanism_Base):
    """Maximize EVC over specified set of control signals for values of monitored states

    Description:
        + Implements EVC maximization (Shenhave et al. 2013), buy:
         [DOCUMENATION HERE]

    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            + kwSystem (System)
            + kwMonitoredStates (list of Mechanisms and/or MechanismOutputStates)

    Class methods:
        None

    Instance attributes:
        system (System):
            System of which EVCMechanism is component, and that it executes to determine the EVC
        controlSignalSearchSpace (list of np.ndarrays):
            list of all combinations of all allocation_sampling_ranges for all ControlSignal Projections
            for all outputStates in self.outputStates;
            each item in the list is an np.ndarray, the dimension of which is the number of self.outputStates
        monitoredValues (3D np.nparray): values of monitored states (self.inputStates) from call of self.executeMethod
        EVCmax (2D np.array):
            values of monitored states (self.inputStates) for EVCmax
        EVCmaxPolicy (1D np.array):
            vector of values (ControlSignal allocations) for EVCmax, one for each outputState in self.outputStates

    Instance methods:
        • validate_params(request_set, target_set, context):
            insure that kwSystem is specified, and validate specifications for monitored states
        • validate_monitored_state(item):
            validate that all specifications for a monitored state are either a Mechanism or MechanismOutputState
        • instantiate_attributes_before_execute_method(context):
            assign self.system and monitoring states (inputStates) specified in kwMonitoredStates
        • instantiate_monitored_states(monitored_states, context):
            parse list of MechanismOutputState(s) and/or Mechanism(s) and call instantiate_monitored_state for each item
        • instantiate_monitored_state(output_state, context):
            extend self.variable to accomodate new inputState
            create new inputState for outputState to be monitored, and assign Mapping Project from it to inputState
        • instantiate_control_signal_projection(projection, context):
            adds outputState, and assigns as sender of to requesting ControlSignal Projection
        • instantiate_execute_method(context):
            construct self.controlSignalSearchSpace from the allocation_sampling_range for the
            ControlSignal Projection associated with each outputState in self.outputStates
        • update(time_scale, runtime_params, context)
            execute System for each combination of controlSignals in self.controlSignalSearchSpace,
                store output values in self.EVCvalues, identify and store maximum in self.EVCmax,
                store the corresponding combination of ControlSignal allocations self.EVCmaxPolicy,
                and assign those allocations to outputState.values
        • execute(params, time_scale, context):
            execute self.system for a combination of controlSignals from self.controlSignalSearchSpace
        • add_monitored_state(state, context):
             validates state as Mechanism or MechanismOutputState specification;
             adds inputState to self.inputStates with Mapping Projection from state
             Note:  used by other objects to add outputState(s) to be monitored by EVC

    """

    functionType = "EVCMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemDefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    from Functions.Utility import LinearCombination
    paramClassDefaults = SystemControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwSystem: None,
                               # MonitoredStates should be a list of MechanismOutputStates (or Mechanisms)
                               #     the values of which are to be monitored
                               kwMonitoredStates: [],
                               # ExecuteMethod and params specifies value aggregation function
                               #     kwWeights should be vector with length = length of kwMonitoredStates
                               kwExecuteMethod: LinearCombination,
                               kwExecuteMethodParams: {kwWeights: [1],
                                                       kwOffset: 0,
                                                       kwScale: 1,
                                                       kwOperation: LinearCombination.Operation.SUM},
                               # CostAggregationFunction specifies how costs are combined across ControlSignals
                               # kwWeight can be added, in which case it should be equal in length
                               #     to the number of outputStates (= ControlSignal Projections)
                               kwCostAggregationFunction:
                                               LinearCombination(
                                                   param_defaults={kwOffset:0,
                                                                   kwScale:1,
                                                                   kwOperation:LinearCombination.Operation.SUM},
                                                   context=functionType+kwCostAggregationFunction),
                               # CostApplicationFunction specifies how aggregated cost is combined with
                               #     aggegated value computed by ExecuteMethod to determine EVC
                               kwCostApplicationFunction:
                                                LinearCombination(
                                                    param_defaults={kwOffset:0,
                                                                    kwScale:1,
                                                                    kwOperation:LinearCombination.Operation.SUM},
                                                    context=functionType+kwCostApplicationFunction)
                               })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
                 # context=NotImplemented):

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType

        self.functionName = self.functionType
        self.controlSignalChannels = OrderedDict()

        super(EVCMechanism, self).__init__(default_input_value=default_input_value,
                                        params=params,
                                        name=name,
                                        prefs=prefs,
                                        context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Validate kwSystem, kwMonitoredStates and kwExecuteMethodParams

        If kwSystem is not specified, raise an exception
        Check that all items in kwMonitoredStates are Mechanisms or MechanismOutputStates for Mechanisms in self.system
        Check that len(kwWeights) = len(kwMonitoredStates)
        """
        super(EVCMechanism, self).validate_params(request_set=request_set, target_set=target_set, context=context)

        if not self.paramsCurrent[kwSystem]:
            raise EVCError("A system must be specified in the kwSystem param to instantiate an EVCMechanism")

        for item in self.paramsCurrent[kwMonitoredStates]:
            self.validate_monitored_state(item, context=context)

        num_weights = len(self.paramsCurrent[kwWeights])
        num_monitored_states = len(self.paramsCurrent[kwMonitoredStates])
        if not num_weights != num_monitored_states:
            raise EVCError("Number of entries ({0}) in kwWeights param for EVC "
                           "does not match the number of monitored states ({1})".
                           format(num_weights, num_monitored_states))

    def validate_monitored_state(self, item, context=NotImplemented):
        """Validate that specification for state to be monitored is either a Mechanism or MechanismOutputState

        Call by both self.validate_params and self.add_monitored_state
        """

        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        if not isinstance(item, (Mechanism, MechanismOutputState)):
            raise EVCError("Specification ({0}) in kwMonitoredStates for EVC of {1} "
                           "that is not a Mechanism or MechanismOutputState ".format(item, self.system.name))
        if isinstance(item, MechanismOutputState):
            item = item.ownerMechanism
        if not item in self.system.terminalMechanisms:
            raise EVCError("Request for EVC of {0} to monitor the outputState(s) of a Mechanism {1}"
                           " that is in a different System ({2})".
                           format(self.system.name, item.name, self.system.name))

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Instantiate self.system and  monitoring state (inputState) for states specified in kwMonitoredStates

        Assign self.system
        If kwMonitoredStates is NOT specified:
            assign an inputState for each outputState of each Mechanism in system.terminalMechanisms
        If kwMonitoredStates IS specified:
            assign an inputState for each MechanismOutState specified
            assign an inputState for all of the outputStates for each Mechanism specified

        """

        self.system = self.paramsCurrent[kwSystem]

        monitored_states = list(self.paramsCurrent[kwMonitoredStates])
        if not monitored_states:
            monitored_states = []
            for mechanism in self.system.terminalMechanisms:
                for state in mechanism.outputStates:
                    monitored_states.append(state)

        self.instantiate_monitored_states(monitored_states, context=context)

    def instantiate_monitored_states(self, monitored_states, context=NotImplemented):
        """Instantiate inputState and Mapping Projections for list of Mechanisms and/or MechanismStates to be monitored

        For each item in monitored_states:
            if it is a MechanismOutputState, call instantiate_monitored_state for item
            if it is a Mechanism, call instantiate_monitored_state for each outputState in Mechanism.outputStates
        """

        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        for item in monitored_states:
            if isinstance(item, MechanismOutputState):
                self.instantiate_monitored_state(item, context=context)
            elif isinstance(item, Mechanism):
                for output_state in item.outputStates:
                    self.instantiate_monitored_state(output_state, context=context)
            else:
                raise EVCError("PROGRAM ERROR: specification ({0}) slipped through that is "
                               "neither a MechanismOutputState nor Mechanism".format(item))

    def instantiate_monitored_state(self, output_state, context=NotImplemented):
        """Instantiate an entry in self.inputStates and a Mapping projection to it from output_state

        Extend self.variable to accomodate new inputState used to monitor output_state
        Instantiate new inputState and add to self.InputStates
        Instantiate Mapping Projection from output_state to new inputState

        Args:
            output_state (MechanismOutputState:
            context:
        """

        state_name = output_state.name + '_Monitor'

        # Extend self.variable to accommodate new inputState
        self.variable = np.append(self.variable, output_state.value)
        variable_item_index = self.variable.size-1

        # Instantiate inputState for output_state to be monitored:
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        input_state = self.instantiate_mechanism_state(
                                        state_type=MechanismInputState,
                                        state_name=state_name,
                                        state_spec=defaultControlAllocation,
                                        constraint_values=np.array(self.variable[variable_item_index]),
                                        constraint_values_name='Default control allocation',
                                        context=context)

        # Instantiate Mapping Projection from output_state to new input_state
        from Functions.Projections.Mapping import Mapping
        Mapping(sender=output_state, receiver=input_state)

        #  Update inputState and inputStates
        try:
            self.inputStates[state_name] = input_state
        except AttributeError:
            self.inputStates = OrderedDict({state_name:input_state})
            self.inputState = list(self.inputStates)[0]

#         # ----------------------------------------
# # FIX: STILL NEEDED HERE??
#         #  Update value by evaluating executeMethod
#         self.update_value()
#         output_item_index = len(self.value)-1

    def instantiate_control_signal_projection(self, projection, context=NotImplemented):
        """Add outputState and assign as sender to requesting ControlSignal Projection

        Assign corresponding outputState
        ?? Register controlSignal range and cost attributes in local attributes

        Args:
            projection:
            context:

        """

        # Call super to instantiate outputStates
        super(EVCMechanism, self).instantiate_control_signal_projection(projection=projection,
                                                                        context=context)

    def instantiate_execute_method(self, context=NotImplemented):
        """Construct controlSignalSearchSpace

        Get allocation_sampling_range for the ControlSignal Projection for each outputState in self.outputStates
        Consruct self.controlSignalSearch (list of np.ndarrays):

        """
        super(EVCMechanism, self).instantiate_execute_method(context=context)

        #  IMPLEMENTATION NOTE: CONSIDER MOVING THIS TO update() METHOD,
        #                      TO BE SURE LATEST VALUES OF allocation_sampling_range ARE USED (IN CASE THEY HAVE CHANGED)
        control_signal_sampling_ranges = []
        # Get allocation_sampling range for all ControlSignal Projections of all outputStates in self.outputStates
        num_output_states = len(self.outputStates)

        for output_state in self.outputStates:
        # for i in range(num_output_states):
            control_signal_sampling_ranges.append(output_state.sendsToProjection.allocationSamples)

#         self.controlSignalSearchSpace = np.matrix([control_signal_sampling_ranges])
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.controlSignalSearchSpace = \
            np.array(np.meshgrid(*control_signal_sampling_ranges)).T.reshape(-1,num_output_states)


    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Search space of control signals for maximum EVC and set value of outputStates accordingly

         Call self.system.execute for each combination of ControlSignals in self.controlSignalSearchSpace
         Store the vector of values for monitored states (inputStates in self.inputStates) for each combination
         Call self.execute to calculate the EVC for each vector, identify the maxium, and assign to self.EVCmax
         Set EVCmaxPolicy to combination of ControlSignal allocations (outputState.values) corresponding to EVCmax
         Set value for each outputState (allocation for each ControlSignal) to the values in self.EVCmaxPolicy
         Return EVCmax

         Note:
         * runtime_params is used for self.execute (that calculates the EVC for each call to system.execute);
             it is NOT used for system.execute — that uses the runtime_params
              provided for the Mechanisms in each Process.congiruation

        Args:
            time_scale:
            runtime_params:
            context:

        Returns (2D np.array): value of outputState for each monitored state (in self.inputStates) for EVCMax
        """

# FIX:  IS THIS THE RIGHT INITIAL VALUE?  OR SHOULD IT BE MAXIMUM NEGATIVE VALUE?
        EVC_current = self.EVCmax = 0


        # Evaluate all combinations of controlSignals (policies)
        for allocation_vector in self.controlSignalSearchSpace: # <-FIX:  THIS NEEDS TO BE CHECKED, PROBABLY CORRECTED
            # Implement the current policy
            for i in range(len(self.outputStates)):
                self.outputStates[i].value = allocation_vector[i]

            # Execute self.system for the current policy
# FIX:  ??PASS IN ANY INPUT?  IF SO, GET FROM SYSTEM??
            self.system.execute(inputs=self.system.inputs, time_scale=time_scale, context=context)

            # Get control cost for this policy
            control_signal_costs = []
            for i in range(len(self.outputStates)):
                # Create vector of controlSignal costs
                control_signal_costs.append(self.outputStates[i].sendsToProjection.cost)
            # Aggregate control costs
            total_current_control_costs = self.paramsCurrent[kwCostAggregationFunction](control_signal_costs)

# FIX:  MAKE SURE self.inputStates AND self.variable IS UPDATED WITH EACH CALL TO system.execute()
# FIX:  IS THIS DONE IN Mechanism.update? DOES THE MEAN NEED TO CALL super().update HERE, AND MAKE SURE IT GETS TO MECHANISM?
# FIX:  self.variable MUST REFLECT VALUE OF inputStates FOR self.execute TO CALCULATE EVC

            # Get value of current policy = weighted sum of values of monitored states
            # Note:  self.variable = value of monitored states (self.inputStates)
            total_current_value = self.execute(params=runtime_params, time_scale=time_scale, context=context)

            # Calculate EVC for the result (default: total value - total cost)
            EVC_current = \
                self.paramsCurrent[kwCostApplicationFunction]([total_current_value, -total_current_control_costs])

            self.EVCmax = max(EVC_current, self.EVCmax)

            # If EVC is greater than the previous value:
            # - store the current set of monitored state value in EVCmaxStateValues
            # - store the current set of controlSignals in EVCmaxPolicy
            if self.EVCmax > EVC_current:
                self.EVCmaxStateValues = self.variable.copy()
                self.EVCmaxPolicy = allocation_vector.copy()

# FIX: ?? NEED TO SET OUTPUT VALUES AND RUN SYSTEM AGAIN?? OR JUST:
# FIX:      - SET values for self.inputStates TO EVCMax ??
# FIX:      - SET values for self.outputStates TO EVCMaxPolicy ??
# FIX:  ??NECESSARY:
        for i in range(len(self.inputStates)):
            self.inputStates[i].value = self.EVCmaxStateValues[i]
        for i in range(len(self.outputStates)):
            self.outputStates[i].value = self.EVCmaxPolicy[i]

        return self.EVCmax

# IMPLEMENTATION NOTE: NOT IMPLEMENTED, AS PROVIDED BY params[kwExecuteMethod]
    # def execute(self, params, time_scale, context):
    #     """Calculate EVC for values of monitored states (in self.inputStates)
    #
    #     Args:
    #         params:
    #         time_scale:
    #         context:
    #     """
    #
    #     return
    #
    #


    def add_monitored_states(self, states_spec, context=NotImplemented):
        """Validate and then instantiate outputStates to be monitored by EVC

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, MechanismOutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used
        All of the outputStates specified must be for a Mechanism that is in self.System

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        self.validate_monitored_state(states_spec, context=context)
        self.instantiate_monitored_states(states_spec, context=context)


