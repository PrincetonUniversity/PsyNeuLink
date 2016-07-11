#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass
from Functions.Mechanisms.SystemControlMechanism import *

from Functions.ShellClasses import *
from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanism_Base


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCMechanism(SystemControlMechanism_Base):
    """Maximize EVC over specified set of control signals for values of monitored states

    Description:
        + Implements EVC maximization (Shenhav et al. 2013)
        [DOCUMENATION HERE:]

        NOTE: self.execute serves as kwValueAggregationFunction
        ALTERNATIVE:  IMPLEMENT FOLLOWING IN paramClassDefaults:
                                       kwValueAggregationFunction:
                                               LinearCombination(
                                                   param_defaults={kwOffset:0,
                                                                   kwScale:1,
                                                                   kwOperation:LinearCombination.Operation.SUM},
                                                   context=functionType+kwValueAggregationFunction),
        # INSTANTIATION:
        # - specification of system:  required param: kwSystem
        # - kwDefaultController:  True =>
        #         takes over all projections from default Controller;
        #         does not take monitored states (those are created de-novo)
        # TBI: - kwControlSignalProjections:
        #         list of projections to add (and for which outputStates should be added)
        # - inputStates: one for each performance/environment variable monitiored


# DOCUMENT:
# 1) Add a predictionMechanism for each origin (input) Mechanism in self.system,
#        and a Process for each pair: [origin, kwIdentityMatrix, prediction]
# 2) Implement self.simulatedSystem that, for each originMechanism
#        replaces Process.inputState with predictionMechanism.value
# 3) Modify EVCMechanism.update() to execute self.simulatedSystem rather than self.system
#    CONFIRM: EVCMechanism.system is never modified in a way that is not reflected in EVCMechanism.simulatedSystem
#                (e.g., when learning is implemented)
# 4) Implement controlSignal allocations for optimal allocation policy in EVCMechanism.system



# NOTE THAT EXCECUTE METHOD ~ ValueAggregationFunction (i.e,. analogous to CostAggregationFunction

# DESCRIBE USE OF MonitoredStatesOptions VS. EXPLICIT SPECIFICADTION OF MECHANISM AND/OR MECHANISMSTATES
# CAN SPECIFIY WEIGHTS IF LIST OF MECHANISMS/ MECHANISMSTATES IS PROVIDED, IN WHICH CASE #WEIGHTS MUST = #STATES SPECIFIED
#              OTHEREWISE (IF MonitoredStatesOptions OR DEFAULT IS USED, WEIGHTS ARE IGNORED

#     kwMonitoredStates must be list of Mechanisms or MechanismOutputStates in Mechanisms that are in kwSystem
#     if Mechanism is specified in kwMonitoredStates, all of its outputStates are used
#     kwMonitoredStates assigns a Mapping Projection from each outputState to a newly created inputState in self.inputStates
#     executeMethod uses LinearCombination to apply a set of weights to the value of each monitored state to compute EVC
#     and then searches space of control signals (using allocationSamples for each) to find combiantion that maxmizes EVC

        #    - wherever a ControlSignal projection is specified, using kwEVC instead of kwControlSignal
        #        this should override the default sender kwSystemDefaultController in ControlSignal.instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_state, parameter_state=NotImplemented) method
        # - specification of executeMethod: default is default allocation policy (BADGER/GUMBY)
        #     constraint:  if specified, number of items in variable must match number of inputStates in kwInputStates
        #                  and names in list in kwMonitor must match those in kwInputStates

#      OBJECTIVE FUNCTION FOR exeuteMethod:
#      Applies linear combination to values of monitored states (self.inputStates)
#      executeMethod is LinearCombination, with weights = linear terms
#      kwExecuteMethodParams = kwWeights
#      Cost is aggregated over controlSignal costs using kwCostAggregationFunction (default: LinearCombination)
            currently, it is specified as an instantiated function rather than a reference to a class
#      Cost is combined with values (agggregated by executeMethod) using kwCostApplicationFunction (default: LinearCombination)
            currently, it is specified as an instantiated function rather than a reference to a class

        # EVALUATION:
        # - evaluation function (as execute method) with one variable item (1D array) for each inputState
        # - mapping projections from monitored states to inputStates
        # - control signal projections established automatically by system implementation (using kwConrolSignal)
        #
        # EXECUTION:
        # - call system.execute for each point in search space
        # - compute evaluation function, and keep track of performance outcomes

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
        predictionMechanisms (list): list of predictionMechanisms added to System for self.system.originMechanisms
        predictionProcesses (list): list of prediction Processes added to System
        controlSignalSearchSpace (list of np.ndarrays):
            list of all combinations of all allocationSamples for all ControlSignal Projections
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
            construct self.controlSignalSearchSpace from the allocationSamples for the
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
    # from Functions.__init__ import DefaultSystem
    paramClassDefaults = SystemControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwSystem: None,
                               # Assigns EVC as DefaultController
                               kwMakeDefaultController:True,
                               # Saves all ControlAllocationPolicies and associated EVC values (in addition to max)
                               kwSaveAllPoliciesAndValues: False,
                               # Can be replaced with a list of MechanismOutputStates or Mechanisms
                               #     the values of which are to be monitored
                               kwMonitoredStates: MonitoredStatesOption.PRIMARY_OUTPUT_STATES,
                               # ExecuteMethod and params specifies value aggregation function
                               kwExecuteMethod: LinearCombination,
                               kwExecuteMethodParams: {kwOffset: 0,
                                                       kwScale: 1,
                                                       # Must be a vector with length = length of kwMonitoredStates
                                                       # kwWeights: [1],
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

# FIX: 7/4/16 MOVE THIS TO SystemControlMechanism (and make sure it works with SystemDefaultControlMechanism
    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Validate kwSystem, kwMonitoredStates and kwExecuteMethodParams

        If kwSystem is not specified, raise an exception
        Check that all items in kwMonitoredStates are Mechanisms or MechanismOutputStates for Mechanisms in self.system
        Check that len(kwWeights) = len(kwMonitorates)
        """

        # MODIFIED 6/28/16 ADDED:
        if not isinstance(request_set[kwSystem], System):
            raise EVCError("A system must be specified in the kwSystem param to instantiate an EVCMechanism")
        self.paramClassDefaults[kwSystem] = request_set[kwSystem]
        # END ADDED

        super(EVCMechanism, self).validate_params(request_set=request_set, target_set=target_set, context=context)

        # Check if kwMonitoredStates param is specified
        try:
            # It IS a MonitoredStatesOption specification
            if isinstance(target_set[kwMonitoredStates], MonitoredStatesOption):
                # Put in a list (standard format for processing by instantiate_monitored_states)
                target_set[kwMonitoredStates] = [target_set[kwMonitoredStates]]
            # It is NOT a MonitoredStatesOption specification, so assume it is a list of Mechanisms or MechanismStates
            else:
                # Validate each item of kwMonitoredStates
                for item in target_set[kwMonitoredStates]:
                    self.validate_monitored_state(item, context=context)
                # Validate kwWeights if it is specified
                try:
                    num_weights = len(target_set[kwExecuteMethodParams][kwWeights])
                except KeyError:
                    # kwWeights not specified, so ignore
                    pass
                else:
                    # Insure that number of weights specified in kwWeights
                    #    equals the number of states instantiated from kwMonitoredStates
                    num_monitored_states = len(target_set[kwMonitoredStates])
                    if not num_weights != num_monitored_states:
                        raise EVCError("Number of entries ({0}) in kwWeights of kwExecuteMethodParam for EVC "
                                       "does not match the number of monitored states ({1})".
                                       format(num_weights, num_monitored_states))
        except KeyError:
            pass


    def validate_monitored_state(self, item, context=NotImplemented):
        """Validate that specification for state to be monitored is either a Mechanism or MechanismOutputState

        Called by both self.validate_params and self.add_monitored_state
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
# DOCUMENT: ADD PREDICTION MECHANISMS
        """Instantiate self.system, inputState(s) specified in kwMonitoredStates, and predictionMechanisms

        Assign self.system
        If kwMonitoredStates is NOT specified:
            assign an inputState for each outputState of each Mechanism in system.terminalMechanisms
        If kwMonitoredStates IS specified:
            assign an inputState for each MechanismOutState specified
            assign an inputState for all of the outputStates for each Mechanism specified
        For each originMechanism in self.system, add a predictionMechanism

        """
        self.system = self.paramsCurrent[kwSystem]

        self.instantiate_monitored_states(context=context)

        # Do this after instantiating monitored_states, so that any predictionMechanisms added
        #    are not incluced in monitored_states (they will be used by EVC to replace corresponding origin Mechanisms)
        self.instantiate_prediction_mechanisms(context=context)

# FIX: INTEGRATE instantiate_monitored_states INTO:
# FIX:     Mechanism.instantiate_mechanism_state_list() AND/OR Mechanism.instantiate_mechanism_state()
# FIX:     (BY PASSING kwMonitoredStates AS state_type ARG AND/OR LIST OF ACTUAL INPUT STATE SPECS IN paramsCurrent[]
# FIX: Move this SystemControlMechanism, and implement relevant versions here and in SystemDefaultControlMechanism
# IMPLEMENT: modify to handle kwMonitoredStatesOption for individual Mechanisms (in SystemControlMechanism):
#                either:  (Mechanism, MonitoredStatesOption) tuple in kwMonitoredStates specification
#                                and/or kwMonitoredStates in individual Mechanism.params[]
# FIX: 7/4/16 Move this SystemControlMechanism, and override with relevant versions here and in SystemDefaultControlMechanism
    def instantiate_monitored_states(self, context=NotImplemented):
        """Instantiate inputState and Mapping Projections for list of Mechanisms and/or MechanismStates to be monitored

        For each item in monitored_states:
            - if it is a MechanismOutputState, call instantiate_monitored_state()
            - if it is a Mechanism, call instantiate_monitored_state for all outputState in Mechanism.outputStates
            - if it is an MonitoredStatesOption specification, initialize monitored_states and implement option

        MonitoredStatesOption is an AutoNumbered Enum declared in SystemControlMechanism
        - It specifies options for assigning outputStates of terminal Mechanisms in the System to monitored_states
        - The options are:
            + PRIMARY_OUTPUT_STATES: assign only the primary outputState for each terminal Mechanism
            + ALL_OUTPUT_STATES: assign all of the outputStates of each terminal Mechanism

        Notes:
        * monitored_states is a list of items that are assigned to the inputStates attribute of a SystemControlMechanism
            it is a convenience variable, and used for coding/descriptive clarity only;
        * all references in comments to "monitored states" can be considered synonymous
            with the inputStates of a SystemControlMechanism

        """

        # Clear self.variable, as items will be assigned in call(s) to instantiate_monitored_state()
        self.variable = None

        # Assign states specified in params[kwMontioredStates] as states to be monitored
        monitored_states = list(self.paramsCurrent[kwMonitoredStates])

        # If specification is a MonitoredStatesOption, store the option and initialize monitored_states as empty list
        if isinstance(monitored_states[0], MonitoredStatesOption):
            option = monitored_states[0]
            monitored_states = []

# FIX:         4) SHOULD DERIVE MONITORED NAME FROM MECHANISM NAME RATHER THAN OUTPUT STATE NAME
            for mechanism in self.system.terminalMechanisms:

                # Assign all outputStates of all terminalMechanisms in system.graph as states to be monitored
                if option is MonitoredStatesOption.PRIMARY_OUTPUT_STATES:
                    # If outputState is already in monitored_states, continue
                    if mechanism.outputState in monitored_states:
                        continue
                    monitored_states.append(mechanism.outputState)

                # Assign all outputStates of all terminalMechanisms in system.graph as states to be monitored
                elif option is MonitoredStatesOption.ALL_OUTPUT_STATES:
                    for state in mechanism.outputStates:
                        if mechanism.outputStates[state] in monitored_states:
                            continue
                        monitored_states.append(mechanism.outputStates[state])

        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        for item in monitored_states:
            if isinstance(item, MechanismOutputState):
                self.instantiate_monitored_state(item, context=context)
            elif isinstance(item, Mechanism):
                for output_state in item.outputStates:
                    self.instantiate_monitored_state(output_state, context=context)
            else:
                raise EVCError("PROGRAM ERROR: outputState specification ({0}) slipped through that is "
                               "neither a MechanismOutputState nor Mechanism".format(item))

        if self.prefs.verbosePref:
            print ("{0} monitoring:".format(self.name))
            for state in monitored_states:
                print ("\t{0}".format(state.name))


    # FIX: Move this SystemControlMechanism, and implement relevant versions here and in SystemDefaultControlMechanism
    def instantiate_monitored_state(self, output_state, context=NotImplemented):
        """Instantiate an entry in self.inputStates and a Mapping projection to from output_state to be monitored

        Extend self.variable to accomodate new inputState used to monitor output_state
        Instantiate new inputState and add to self.InputStates
        Instantiate Mapping Projection from output_state of monitored state to new self.inputState[i]

        Args:
            output_state (MechanismOutputState:
            context:
        """

        state_name = output_state.name + '_Monitor'

        # Extend self.variable to accommodate new inputState
        if self.variable is None:
            self.variable = np.atleast_2d(output_state.value)
        else:
            self.variable = np.append(self.variable, np.atleast_2d(output_state.value), 0)
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
        test = True

    def instantiate_prediction_mechanisms(self, context=NotImplemented):
        """Add prediction Process for each origin (input) Mechanism in System

        Args:
            context:
        """

        from Functions.Mechanisms.AdaptiveIntegrator import AdaptiveIntegratorMechanism
        from Functions.Process import Process_Base

        # Instantiate a predictionMechanism for each origin (input) Mechanism in self.system,
        #    instantiate a Process (that maps the origin to the prediction mechanism),
        #    and add that Process to System.processes list
        self.predictionMechanisms = []
        self.predictionProcesses = []
        for mech in self.system.originMechanisms:

            # Instantiate prediction mechanism using AdaptiveIntegratorMechanism
            # IMPLEMENTATION NOTE: SHOULD MAKE THIS A PARAMETER (kwPredictionMechanism) OF EVCMechanism
            prediction_mechanism = AdaptiveIntegratorMechanism(name=mech.name + "_" + kwPredictionMechanism)
            self.predictionMechanisms.append(prediction_mechanism)
            # Assign origin and associated prediction mechanism (with same phase as origin Mechanism) to a Process
            prediction_process = Process_Base(default_input_value=NotImplemented,
                                              params={
                                                  kwConfiguration:[(mech, mech.phaseSpec),
                                                                   kwIdentityMatrix,
                                                                   (prediction_mechanism, mech.phaseSpec)]},
                                              name=mech.name + "_" + kwPredictionProcess
                                              )
            # Add the process to the system's list of processes, and the controller's list of prediction processes
            self.system.processes.append((prediction_process, None))
            self.predictionProcesses.append(prediction_process)

        # Re-instantiate System.graph with predictionMechanism Processes added
        # CONFIRM THAT self.system.variable IS CORRECT BELOW:
        self.system.instantiate_graph(self.system.variable, context=context)
# FIX: ADD INPUTS TO EVC-GENERATED PROCESSES (FROM PREDICTION MECHANISMS) ??IN EVC.instantiate_prediction_mechanisms??
        # Replace origin mechanisms with Prediction mechanisms as monitored states and/or inputs to System
        # ?? Add value of predictions mechanisms as inputs to new prediction Processes

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Construct and search space of control signals for maximum EVC and set value of outputStates accordingly

        Get allocationSamples for the ControlSignal Projection for each outputState in self.outputStates
        Construct self.controlSignalSearchSpace (2D np.array, each item of which is a permuted set of samples):
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

        #region CONSTRUCT SEARCH SPACE
        # IMPLEMENTATION NOTE: MOVED FROM instantiate_execute_method
        #                      TO BE SURE LATEST VALUES OF allocationSamples ARE USED (IN CASE THEY HAVE CHANGED)
        #                      SHOULD BE PROFILED, AS MAY BE INEFFICIENT TO EXECUTE THIS FOR EVERY RUN
        control_signal_sampling_ranges = []
        # Get allocationSamples for all ControlSignal Projections of all outputStates in self.outputStates
        num_output_states = len(self.outputStates)

        for output_state in self.outputStates:
            for projection in self.outputStates[output_state].sendsToProjections:
                control_signal_sampling_ranges.append(projection.allocationSamples)

        # Construct controlSignalSearchSpace:  set of all permutations of ControlSignal allocations
        #                                     (one sample from the allocationSample of each ControlSignal)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.controlSignalSearchSpace = \
            np.array(np.meshgrid(*control_signal_sampling_ranges)).T.reshape(-1,num_output_states)
        # END MOVE
        #endregion

        #region ASSIGN SIMULATION INPUT(S)
        # For each prediction mechanism, assign its value as input to corresponding process for the simulation
        for mech in self.predictionMechanisms:
            # Get the Process.inputState for the origin Mechanism corresponding to mech (current predictionMechanism)
            #    and set its value to value of predictionMechanism
            for input_state_name, input_state in list(mech.inputStates.items()):
                for projection in input_state.receivesFromProjections:
                    # # TEST:
                    # mech.value = np.atleast_1d(5)
                    projection.sender.ownerMechanism.inputState.receivesFromProjections[0].sender.value = mech.value
                    # TEST = True
        #endregion

        #region RUN SIMULATION

        self.EVCmax = 0 # <- FIX:  IS THIS THE RIGHT INITIAL VALUE?  OR SHOULD IT BE MAXIMUM NEGATIVE VALUE?
        self.EVCvalues = []
        self.EVCpolicies = []

        # Reset context so that System knows this is a simulation (to avoid infinitely recursive loop)
        context = context.replace('EXECUTING', '{0} {1}'.format(self.name, kwEVCSimulation))

        if self.prefs.reportOutputPref:
            progress_bar_rate_str = ""
            search_space_size = len(self.controlSignalSearchSpace)
            progress_bar_rate = int(10 ** (np.log10(search_space_size)-2))
            if progress_bar_rate > 1:
                progress_bar_rate_str = str(progress_bar_rate) + " "
            print("\n{0} evaluating EVC for {1} (one dot for each {2}of {3} samples): ".
                  format(self.name, self.system.name, progress_bar_rate_str, search_space_size))

        # Evaluate all combinations of controlSignals (policies)
        sample = 0
        self.EVCmaxStateValues = self.variable.copy()
        self.EVCmaxPolicy = self.controlSignalSearchSpace[0] * 0.0

        for allocation_vector in self.controlSignalSearchSpace:

            if self.prefs.reportOutputPref:
                increment_progress_bar = (progress_bar_rate < 1) or not (sample % progress_bar_rate)
                if increment_progress_bar:
                    print(kwProgressBarChar, end='', flush=True)
            sample +=1

            # Implement the current policy over ControlSignal Projections
            for i in range(len(self.outputStates)):
                list(self.outputStates.values())[i].value = np.atleast_1d(allocation_vector[i])

            # Execute self.system for the current policy
# FIX: NEED TO CYCLE THROUGH PHASES, AND COMPUTE VALUE FOR RELEVANT ONES (ALWAYS THE LAST ONE??)
# FIX: NEED TO APPLY predictionMechanism.value AS INPUT TO RELEVANT MECHANISM IN RELEVANT PHASE
            for i in range(self.system.phaseSpecMax):
                CentralClock.time_step = i
                self.update_input_states(runtime_params=runtime_params,time_scale=time_scale,context=context)
                self.system.execute(inputs=self.system.inputs, time_scale=time_scale, context=context)

            # Get control cost for this policy
            # Iterate over all outputStates (controlSignals)
            for i in range(len(self.outputStates)):
                # Get projections for this outputState
                output_state_projections = list(self.outputStates.values())[i].sendsToProjections
                # Iterate over all projections for the outputState
                for projection in output_state_projections:
                    # Get ControlSignal cost
                    control_signal_cost = np.atleast_2d(projection.cost)
                    # Build vector of controlSignal costs
                    if i==0:
                        control_signal_costs = np.atleast_2d(control_signal_cost)
                    else:
                        control_signal_costs = np.append(control_signal_costs, control_signal_cost, 0)

            # Aggregate control costs
            total_current_control_costs = self.paramsCurrent[kwCostAggregationFunction].execute(control_signal_costs)

# FIX:  MAKE SURE self.inputStates AND self.variable IS UPDATED WITH EACH CALL TO system.execute()
# FIX:  IS THIS DONE IN Mechanism.update? DOES THE MEAN NEED TO CALL super().update HERE, AND MAKE SURE IT GETS TO MECHANISM?
# FIX:  self.variable MUST REFLECT VALUE OF inputStates FOR self.execute TO CALCULATE EVC

            variable = []
            for input_state in list(self.inputStates.values()):
                variable.append(input_state.value)
            variable = np.atleast_2d(variable)

            # Get value of current policy = weighted sum of values of monitored states
            # Note:  self.variable = value of monitored states (self.inputStates)
            total_current_value = self.execute(variable=self.variable,
                                               params=runtime_params,
                                               time_scale=time_scale,
                                               context=context)

            # Calculate EVC for the result (default: total value - total cost)
            EVC_current = \
                self.paramsCurrent[kwCostApplicationFunction].execute([total_current_value, -total_current_control_costs])
            self.EVCmax = max(EVC_current, self.EVCmax)

            # Add to list of EVC values and allocation policies if save option is set
            if self.paramsCurrent[kwSaveAllPoliciesAndValues]:
                self.EVCvalues.append(EVC_current)
                self.EVCpolicies.append(allocation_vector.copy())

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
        if self.prefs.reportOutputPref:
            print("\nEVC simulation completed")
#endregion

        #region ASSIGN CONTROL SIGNALS

        # Assign allocations to controlSignals (self.outputStates) for optimal allocation policy:
        for i in range(len(self.outputStates)):
            list(self.outputStates.values())[i].value = np.atleast_1d(self.EVCmaxPolicy[i])

        # Assign max values for optimal allocation policy to self.inputStates (for reference only)
        for i in range(len(self.inputStates)):
            list(self.inputStates.values())[i].value = np.atleast_1d(self.EVCmaxStateValues[i])

        # Report EVC max info
        if self.prefs.reportOutputPref:
            print ("\nMaximum EVC for {0}: {1}".format(self.system.name, float(self.EVCmax)))
            print ("ControlSignal allocations for maximum EVC:")
            for i in range(len(self.outputStates)):
                print("\t{0}: {1}".format(list(self.outputStates.values())[i].name,
                                        self.EVCmaxPolicy[i]))
            print()
        #endregion

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

    def inspect(self):

        print ("\n{0} is monitoring the following mechanism outputStates:".format(self.name))
        for state_name, state in list(self.inputStates.items()):
            for projection in state.receivesFromProjections:
                print ("\t{0}: {1}".format(projection.sender.ownerMechanism.name, projection.sender.name))

        print ("\n{0} is controlling the following mechanism parameters:".format(self.name))
        for state_name, state in list(self.outputStates.items()):
            for projection in state.sendsToProjections:
                print ("\t{0}: {1}".format(projection.receiver.ownerMechanism.name, projection.receiver.name))
