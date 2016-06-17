#
# *********************************************  Process ***************************************************************
#

import re
import Functions
from Functions.ShellClasses import *
from Globals.Registry import register_category
from Functions.Mechanisms.Mechanism import Mechanism_Base
from Functions.Mechanisms.Mechanism import mechanism

# *****************************************    PROCESS CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# Labels for items in configuration entry tuples
MECHANISM = 0
EXECUTE_METHOD_PARAM_SPECS = 1

ProcessRegistry = {}

kwProcessInputState = 'ProcessInputState'
from Functions.MechanismStates.MechanismOutputState import MechanismOutputState

# DOCUMENT:  HOW DO MULTIPLE PROCESS INPUTS RELATE TO # OF INPUTSTATES IN FIRST MECHANISM
#            WHAT HAPPENS IF LENGTH OF INPUT TO PROCESS DOESN'T MATCH LENGTH OF VARIABLE FOR FIRST MECHANISM??

class ProcessInputState(MechanismOutputState):
    """Represent input to process and provide to first Mechanism in Configuration

    Each instance encodes an item of the Process input (one of the 1D arrays in the 2D np.array input) and provides
        the input to a Mapping projection to one or more inputStates of the first Mechanism in the Configuration;
        see Process Description for mapping when there is more than one Process input value and/or Mechanism inputState

     Notes:
      * Declared as sublcass of MechanismOutputState so that it is recognized as a legitimate sender to a Projection
           in Projection.instantiate_sender()
      * self.value is used to represent input to Process provided as variable arg on command line

    """
    def __init__(self, owner=None, variable=NotImplemented, prefs=NotImplemented):
        """Pass variable to mapping projection from Process to first Mechanism in Configuration

        :param variable:
        """
        self.name = owner.name + "_" + kwProcessInputState
        self.prefs = prefs
        self.sendsToProjections = []
        self.ownerMechanism = owner
        self.value = variable


class ProcessError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


class Process_Base(Process):
    """Implement abstract class for Process category of Function class, used by ControlSignal and Mapping subclass types

    Description:
        A process is defined by a kwConfiguration param (list of mechanisms) and a time scale.  Executing a process
         executes the mechanisms in the order that they appear in the configuration.

    Instantiation:
        Processes can be instantiated in one of two ways:
            - by a direct call to Process()
            [TBI: - in a list to a call to System()]
        Process instantiates its configuration by assigning:
            - a projection from the Process to the inputState of the first mechanism in the configuration
            - a projection from each mechanism in the list to the next (if one is not already specified)
            - any params specified as part of a (mechanism, params) tuple in the configuration
            - Process.outputState to Mechanism.outputState of the last mechanism in the configuration

    Initialization arguments:
        - input (value): used as input to first Mechanism in Configuration (default: first mech variableInstanceDefault)
            ProcessInputState is instantiated to represent each item in the input, and generate a mapping projection
                for it to the first Mechanism in the Configuration:
                if len(Process.input) == len(Mechanism.variable):
                    - create one projection for each of the Mechanism.inputState(s)
                if len(Process.input) == 1 but len(Mechanism.variable) > 1:
                    - create a projection for each of the Mechanism.inputStates, and provide Process.input.value to each
                if len(Process.input) > 1 but len(Mechanism.variable) == 1:
                    - create one projection for each Process.input value and assign all to Mechanism.inputState
                otherwise,  if len(Process.input) != len(Mechanism.variable) and both > 1:
                    - raise exception:  ambiguous mapping from Process input values to first Mechanism's inputStates
        - params (dict):
            kwConfiguration (list): (default: single Mechanism_Base.defaultMechanism)
                Each config_entry must be one of the following, that is used to instantiate the mechanisms in the list:
                    + Mechanism object
                    + Mechanism type (class) (e.g., DDM)
                    + descriptor keyword for a Mechanism type (e.g., kwDDM)
                    + specification dict for Mechanism; the dict can have the following entries (see Mechanism):
                        + kwMechanismType (Mechanism subclass): if absent, Mechanism_Base.defaultMechanism is used
                        + entries with keys = standard args of Mechanism.__init__:
                            "input_template":<value>
                            kwParamsArg:<dict>
                                kwExecuteMethodParams:<dict>
                            kwNameArg:<str>
                            kwPrefsArg"prefs":<dict>
                            kwContextArg:<str>
                    Notes:
                    * specification of any of the params above are used for instantiation of the corresponding mechanism
                         (i.e., its paramInstanceDefaults), but NOT its execution;
                    * runtime params can be passed to the Mechanism (and its states and projections) using a tuple:
                        + (Mechanism, dict):
                            Mechanism can be any of the above
                            dict: can be one (or more) of the following:
                                + kwMechanismInputStateParams:<dict>
                                + kwMechanismParameterStateParams:<dict>
                           [TBI + kwMechanismOutputStateParams:<dict>]
                                - each dict will be passed to the corresponding MechanismState
                                - params can be any permissible executeParamSpecs for the corresponding MechanismState
                                - dicts can contain the following embedded dicts:
                                    + kwExecuteMethodParams:<dict>:
                                         will be passed the MechanismState's execute method,
                                             overriding its paramInstanceDefaults for that call
                                    + kwProjectionParams:<dict>:
                                         entry will be passed to all of the MechanismState's projections, and used by
                                         by their execute methods, overriding their paramInstanceDefaults for that call
                                    + kwMappingParams:<dict>:
                                         entry will be passed to all of the MechanismState's Mapping projections,
                                         along with any in a kwProjectionParams dict, and override paramInstanceDefaults
                                    + kwControlSignalParams:<dict>:
                                         entry will be passed to all of the MechanismState's ControlSignal projections,
                                         along with any in a kwProjectionParams dict, and override paramInstanceDefaults
                                    + <projectionName>:<dict>:
                                         entry will be passed to the MechanismState's projection with the key's name,
                                         along with any in the kwProjectionParams and Mapping or ControlSignal dicts

        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): used to track object/class assignments and methods in hierarchy

        NOTES:
            * if no configuration or time_scale is provided:
                a single mechanism of Mechanism class default mechanism and TRIAL are used
            * process.input is set to the inputState.value of the first mechanism in the configuration
            * process.output is set to the outputState.value of the last mechanism in the configuration

    ProcessRegistry:
        All Processes are registered in ProcessRegistry, which maintains a dict for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Processes can be named explicitly (using the name='<name>' argument).  If this argument is omitted,
        it will be assigned "Mapping" with a hyphenated, indexed suffix ('Mapping-n')

    Execution:
        - Process.execute calls mechanism.update_states_and_execute for each mechanism in its configuration in sequence
            • input specified as arg in execution of Process is provided as input to first mechanism in configuration
            • output of last mechanism in configuration is assigned as Process.ouputState.value
            • SystemDefaultController is executed before execution of each mechanism in the configuration
            • notes:
                * the same mechanism can be listed more than once in a configuration, inducing recurrent processing
                * if it is the first mechanism, it will receive its input from the Process only once (first execution)
                [TBI: add option to allow input to be provided every time mechanism it executed]
            • Process.ouputState.value receives Mechanism.outputState.value of last mechanism in the configuration

    Class attributes:
        + functionCategory (str): kwProcessFunctionCategory
        + className (str): kwProcessFunctionCategory
        + suffix (str): " <kwMechanismFunctionCategory>"
        + registry (dict): ProcessRegistry
        + classPreference (PreferenceSet): ProcessPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault = inputValueSystemDefault                     # Used as default input value to Process)
        + paramClassDefaults = {kwConfiguration: [Mechanism_Base.defaultMechanism],
                                kwTimeScale: TimeScale.TRIAL}

    Class methods:
        • execute(input, control_signal_allocations, time_scale):
            executes the process by calling execute_functions of the mechanisms (in order) in the configuration list
            assigns input to sender.output (and passed through mapping) of first mechanism in the configuration list
            assigns output of last mechanism in the configuration list to self.output
            returns output after either one time_step or the full trial (determined by time_scale)
        • get_configuration(): returns configuration (list)
        • get_mechanism_dict(): returns mechanismDict (dict)
        • register_process(): registers process with ProcessRegistry
        [TBI: • adjust(control_signal_allocations=NotImplemented):
            modifies the control_signal_allocations while the process is executing;
            calling it without control_signal_allocations functions like interrogate
            returns (responseState, accuracy)
        [TBI: • interrogate(): returns (responseState, accuracy)
        [TBI: • terminate(): terminates the process and returns output
        [TBI: • accuracy(target):
            a function that uses target together with the configuration's output.value(s)
            and its accuracyFunction to return an accuracy measure;
            the target must be in a configuration-appropriate format (checked with call)

    Instance attributes:
        + configuration (list): set in params[kwConfiguration]
            an ordered list of tuples that defines how the process is carried out;
                (default: the default mechanism for the Mechanism class — currently: DDM)
                Note:  this is constructed from the kwConfiguration param, which may or may not contain tuples;
                       all entries of kwConfiguration param are converted to tuples for self.configuration
                       for entries that are not tuples, None is used for the param (2nd) item of the tuple
        + sendsToProjections (list)           | these are used to instantiate a projection from Process
        + ownerMechanism (None)               | to first mechanism in the configuration list
        + value (value)                       | value and executeMethodDefault are used to specify input to Process
        + executeMethodOutputDefault (value)  | they are zeroed out after executing the first item in the configuration
        + outputState (MechanismsState object) - reference to MechanismOutputState of last mechanism in configuration
            updated with output of process each time process.execute is called
        + timeScale (TimeScale): set in params[kwTimeScale]
             defines the temporal "granularity" of the process; must be of type TimeScale
                (default: TimeScale.TRIAL)
        + mechanismDict (dict) - dict of mechanisms used in configuration (one config_entry per mechanism type):
            - key: mechanismName
            - value: mechanism
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying ProcessPreferenceSet


    Instance methods:
        None
    """
    functionCategory = kwProcessFunctionCategory
    className = functionCategory
    suffix = " " + className
    functionType = "Process"

    registry = ProcessRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    # Use inputValueSystemDefault as default input to process
    variableClassDefault = inputValueSystemDefault

    paramClassDefaults = Function.paramClassDefaults.copy()
    # paramClassDefaults.update({kwConfiguration: [DefaultMechanism],
    paramClassDefaults.update({kwConfiguration: [Mechanism_Base.defaultMechanism],
                               kwTimeScale: TimeScale.TRIAL
                               })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign category-level preferences, register category, call super.__init__ (that instantiates configuration)


        :param default_input_value:
        :param params:
        :param name:
        :param prefs:
        :param context:
        """

        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name
        self.functionName = self.functionType
        self.configuration = NotImplemented
        self.mechanismDict = {}
        register_category(self, Process_Base, ProcessRegistry, context=context)

        if context is NotImplemented:
            # context = self.__class__.__name__
            context = kwInit + self.name

        super(Process_Base, self).__init__(variable_default=default_input_value,
                                           param_defaults=params,
                                           name=self.name,
                                           prefs=prefs,
                                           context=context)
        if self.prefs.reportOutputPref:
            print("\n{0} initialized with:\n- configuration: [{1}]".
                  format(self.name, self.configurationMechanismNames.__str__().strip("[]")))

    def validate_variable(self, variable, context=NotImplemented):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        :param variable:
        :param context:
        :return:
        """

        super(Process_Base, self).validate_variable(variable, context)

        # Force Process variable specification to be a 2D array (to accommodate multiple input states of 1st mech):
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Call methods that must be run before execute method is instantiated

        Need to do this before instantiate_execute_method as mechanisms in configuration must be instantiated
            in order to assign input projection and self.outputState to first and last mechanisms, respectively

        :param context:
        :return:
        """
        self.instantiate_configuration(context=context)
        # super(Process_Base, self).instantiate_execute_method(context=context)

    def instantiate_execute_method(self, context=NotImplemented):
        """Override Function.instantiate_execute_method:

        This is necessary to:
        - insure there is no kwExecuteMethod specified (not allowed for a Process object)
        - suppress validation (and attendant execution) of Process execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in the configuration have already been validated

        :param context:
        :return:
        """

        if self.paramsCurrent[kwExecuteMethod] != self.execute:
            print("Process object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[kwExecuteMethod], kwExecuteMethod)
            self.paramsCurrent[kwExecuteMethod] = self.execute
        # If validation pref is set, instantiate and execute the process (including all mechanisms in the configuration)
        if self.prefs.paramValidationPref:
            super(Process_Base, self).instantiate_execute_method(context=context)
        # Otherwise, just set Process output info to the corresponding info for the last mechanism in the configuration
        else:
            self.value = self.configuration[-1][MECHANISM].outputState.value

# DOCUMENTATION:
#         Uses paramClassDefaults[kwConfiguration] == [Mechanism_Base.defaultMechanism] as default
# FIX:
#     ** PROBLEM: self.value IS ASSIGNED TO variableInstanceDefault WHICH IS 2D ARRAY, BUT PROJECTION EXECUTION FUNCTION TAKES 1D ARRAY
#         Assign projection from Process (self.value) to inputState of the first mechanism in the configuration
#     **?? WHY DO THIS, IF SELF.VALUE HAS BEEN ASSIGNED AN INPUT VALUE, AND PROJECTION IS PROVIDING INPUT TO MECHANISM??
#         Assigns variableInstanceDefault to variableInstanceDefault of first mechanism in configuration
    def instantiate_configuration(self, context):
        """Construct configuration list of Mechanisms used to run process

        Assign projections from each Mechanism to the next one in the list, if it doesn't already have one
        Assign ouputState for last Mechanism in list to Process.outputState
        Create list of names
        Add Mechanisms to mechanismDict

        :param context:
        :return:
        """

        context = context + kwInstantiate + kwConfiguration
        configuration = self.paramsCurrent[kwConfiguration]
        # Initialize list of execute params
        self.executeMethodParams = [None] * len(configuration)

        #region ITERATE THROUGH CONFIGURATION
        # Go through list of mechanisms in configuration, validate, and for all but first mechanism:
        # - parse each entry, making it a (mechanism, params) tuple if it is not (for further processing)
        # - assign default projection from previous mechanism's outputState to current one's inputState;
        #   (first mechanism gets projection from Process self.value - see below)
        i=0 # Can't use index based on config_entry (i.e., names), since there may be repeats

        for config_entry in configuration:

            #region PARSE CONFIGURATION ENTRY
            # - each should be either an "exposed" Mechanism specification or a (Mechanism,Params) tuple
            # - if not already a tuple, convert to one and add None for params item:
            if not isinstance(configuration[i], tuple):
                configuration[i] = (configuration[i], None)

            mech, params = configuration[i]

            if params and not isinstance(params, dict):
                raise PermissionError("Params entry ({0}) of tuple in item {1} of process configuration is not a dict".
                                      format(params, i))
            #endregion

            #region INSTANTIATE MECHANISM
            # If mechanism item of the tuple is not a Mechanism object
            #    instantiate specification, and replace tuple mechanism entry with Mechanism object
            if not isinstance(mech, Mechanism):
                mech = mechanism(mech, context=context)
                if not mech:
                    raise ProcessError("Entry {0} ({1}) is not a recognized form of Mechanism specification".
                                       format(i, config_entry))
                configuration[i] = (mech, configuration[i][EXECUTE_METHOD_PARAM_SPECS])
            #endregion

            #region INSTANTIATE PROJECTION(S) TO FIRST MECHANISM

            # For first Mechanism in configuration:
            # - initialize configurationMechanismNames
            # - assign input(s) to Mechanism.inputState(s)
            if i == 0:
                # Set up list of Mechanism names for Configuration
                self.configurationMechanismNames = [mech.name]
                self.assign_process_inputs(mech)
            #endregion

            #region PROCESS SUBSEQUENT MECHANISMS
            # - add to configurationMechanismNames
            # - if it doesn't already have a projection from preceding mechanism in list, create and add it
            else:
                self.configurationMechanismNames.append(mech.name)
                preceding_mech = configuration[i-1][MECHANISM]

                # If preceding mechanism is not the sender of any projections received by the current one's inputState
                if not (any(preceding_mech == projection.sender.ownerMechanism
                            for projection in mech.inputState.receivesFromProjections)):
                    # Instantiate mapping projection from preceding mechanism to current one:
                    from Functions.Projections.Mapping import Mapping
                    Mapping(sender=preceding_mech, receiver=mech)
                    if self.prefs.verbosePref:
                        print("Mapping projection added from mechanism {0} to mechanism {1} in configuration of {2}".
                              format(preceding_mech.name, mech.name, self.name))
            #endregion

            # Add mechanism to process.mechanismsDict if it is not already there
            self.mechanismDict.setdefault(mech.name, mech)

            i+=1
#endregion


        #region SET CONFIG PARAMS AND STATE VARIABLES
        self.configuration = configuration
        self.firstMechanism = self.configuration[0][MECHANISM]
        self.lastMechanism = self.configuration[-1][MECHANISM]

        # Set variableInstanceDefault to one for first mechanism in configuration
        # Notes:
        # * need to do this here (rather than validate_variable),
        #       so that it is after configuration has been processed (and first_mechanism is assigned)
        # * no need for further validation, since the execute method for a Process
        #    is simply to pass its input to the first mechanism in the configuration
        # * this will be a 2D np.array (since Mechanisms require this as their variable format)
        # MODIFIED 6/15/16 COMMENTED OUT:
        # self.variableInstanceDefault = self.firstMechanism.variableInstanceDefault

        # Assign process outputState to last mechanisms in configuration
        self.outputState = self.lastMechanism.outputState
        #endregion

    def assign_process_inputs(self, mech):
        """Create projection(s) for each Process input item to inputState(s) of the first Mechanism in Configuration

        For each item in Process input:
        - create process_input_state, as sender for Mapping Projection to Mechanism inputState
        - create the Mapping projection (with process_input_state as sender, and Mechanism as receiver)

        If len(Process.input) == len(Mechanism.variable):
            - create one projection for each of the Mechanism.inputState(s)
        If len(Process.input) == 1 but len(Mechanism.variable) > 1:
            - create a projection for each of the Mechanism.inputStates, and provide Process.input.value to each
        If len(Process.input) > 1 but len(Mechanism.variable) == 1:
            - create one projection for each Process.input value and assign all to Mechanism.inputState
        Otherwise,  if len(Process.input) != len(Mechanism.variable) and both > 1:
            - raise exception:  ambiguous mapping from Process input values to first Mechanism's inputStates

        :param mech:
        :return:
        """

        # Convert Process input to 2D np.array
        process_input = convert_to_np_array(self.variable,2)

        # Get number of Process inputs
        num_process_inputs = len(process_input)

        # Get number of Mechanism.inputStates
        #    - assume mech.variable is a 2D np.array, and that
        #    - there is one inputState for each item in Mechanism.variable
        num_mech_input_states = len(mech.variable)

        # There is a mismatch between number of Process inputs and number of Mechanism.inputStates:
        if num_process_inputs > 1 and num_mech_input_states > 1 and num_process_inputs != num_mech_input_states:
            raise ProcessError("Mismatch between number of input values ({0}) for {1} and "
                               "number of inputStates ({2}) for {3}".format(num_process_inputs,
                                                                            self.name,
                                                                            num_mech_input_states,
                                                                            mech.name))
        # Create a list of Process input states
        self.process_input_states = []
        # Create input state for each item of Process input, and assign to list
        for i in range(num_process_inputs):
            process_input_state = ProcessInputState(owner=self,
                                                    variable=process_input[i],
                                                    prefs=self.prefs)
            self.process_input_states.append(process_input_state)

        from Functions.Projections.Mapping import Mapping

        # If there is the same number of Process input values and Mechanism.inputStates, assign one to each
        if num_process_inputs == num_mech_input_states:
            for i in range(num_mech_input_states):
                # Insure that each Process input value is compatible with corresponding variable of Mechanism.inputState
                if not iscompatible(process_input[i], mech.variable[i]):
                    raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                       "variable for corresponding inputState of {3}".
                                       format(i, process_input[i], self.name, mech.name))
                # Create Mapping projection from Process input state to corresponding Mechanism.inputState
                Mapping(sender=self.process_input_states[i], receiver=list(mech.inputStates.items())[i][1])
                if self.prefs.verbosePref:
                    print("Assigned input value {0} ({1}) of {2} to corresponding inputState of {3}".
                          format(i, process_input[i], self.name, mech.name))

        # If the number of Process inputs and Mechanism.inputStates is unequal, but only a single of one or the other
        # - if there is a single Process input value and multiple Mechanism.inputStates,
        #     instantiate a single Process input state with projections to each of the Mechanism.inputStates
        # - if there are multiple Process input values and a single Mechanism.inputState,
        #     instantiate multiple Process input states each with a projection to the single Mechanism.inputState
        else:
            for i in range(num_mech_input_states):
                for j in range(num_process_inputs):
                    if not iscompatible(process_input[j], mech.variable[i]):
                        raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                           "variable ({3}) for inputState {4} of {5}".
                                           format(j, process_input[j], self.name,
                                                  mech.variable[i], i, mech.name))
                    # Create Mapping projection from Process buffer_intput_state to corresponding Mechanism.inputState
                    Mapping(sender=self.process_input_states[j], receiver=list(mech.inputStates.items())[i][1])
                    if self.prefs.verbosePref:
                        print("Assigned input value {0} ({1}) of {2} to inputState {3} of {4}".
                              format(j, process_input[j], self.name, i, mech.name))

    def execute(self,
                input=NotImplemented,
                time_scale=NotImplemented,
                runtime_params=NotImplemented,
                context=NotImplemented
                ):
        """Coordinate execution of mechanisms in project list (self.configuration)

        First check that input is provided (required)
        Then go through mechanisms in configuration list, and execute each one in the order they appear in the list

        ** MORE DOCUMENTATION HERE:  ADDRESS COORDINATION ACROSS PROCESSES (AT THE LEVEL OF MECHANISM) ONCE IMPLEMENTED

        Arguments:
# DOCUMENT:
        - input (list of numbers): input to mechanism;
            must be consistent with self.input type definition for the receiver.input of
                the first mechanism in the configuration list
        - time_scale (TimeScale enum): determines whether mechanisms are executed for a single time step or a trial
        - params (dict):  set of params defined in paramClassDefaults for the subclass
        - context (str): not currently used

        Returns: output of process (= output of last mechanism in configuration)

        IMPLEMENTATION NOTES:
         Still need to:
         * coordinate execution of multiple processes (in particular, mechanisms that appear in more than one process)
         * deal with different time scales

        :param input (list of numbers): (default: variableInstanceDefault)
        :param time_scale (TimeScale): (default: TRIAL)
        :param params (dict):
        :param context (str):
        :return output (list of numbers):
        """

        if context is NotImplemented:
            context = kwExecuting + self.name

        # Validate input
        if input is NotImplemented:
            input = self.variableInstanceDefault
            if (self.prefs.verbosePref and
                    not (context is NotImplemented or kwFunctionInit in context)):
                print("- No input provided;  default will be used: {0}")
        else:
            input = convert_to_np_array(input, 2)


        # Assign items in input to value of each process_input_state
        for i in range (len(self.process_input_states)):
            self.process_input_states[i].value = input[i]

        self.check_args(input,runtime_params)

        if time_scale is NotImplemented:
            self.timeScale = TimeScale.TRIAL

        if (kwExecuting in context):  # Note: not necessarily so, as execute method is also called for validation
            if self.prefs.reportOutputPref:
                print("\n{0} executing with:\n- configuration: [{1}]".
                      format(self.name, re.sub('[\[,\],\n]','',str(self.configurationMechanismNames))))

        # Use value of Process as input to first Mechanism in Configuration
        self.variable = input

        # Report input if reporting preference is on and this is not an initialization run
        if self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context):
            print("- input: {1}".format(self.name, re.sub('[\[,\],\n]','',str(self.variable))))

        #region EXECUTE EACH MECHANISM
        # Execute each mechanism in the configuration, in the order listed
        i = 0 # Need to use this, as can't use index on mechanism since it may be repeated in the configuration
        for mechanism, params in self.configuration:

            CentralClock.time_step = i

            # Note:  DON'T call include input arg, as that will be resolved by mechanism from its sender projections
            mechanism.update(time_scale=self.timeScale,
                             runtime_params=params,
                             context=context)
            # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR THE MECHANISM ITSELF
            if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
                print("\n{0} executed {1}:\n- output: {2}".format(self.name,
                                                                  mechanism.name,
                                                                  re.sub('[\[,\],\n]','',
                                                                         str(mechanism.outputState.value))))
            if not i:
                # Zero input to first mechanism after first run (in case it is repeated in the configuration)
                # IMPLEMENTATION NOTE:  in future version, add option to allow Process to continue to provide input
                self.variable = self.variable * 0
            i += 1
        #endregion

        if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
            print("\n{0} completed:\n- output: {1}".format(self.name,
                                                           re.sub('[\[,\],\n]','',str(self.outputState.value))))

        return self.outputState.value

    def get_configuration(self):
        """Return configuration (list of Projection tuples)
        The configuration is an ordered list of Project tuples, each of which contains:
             sender (MechanismState object)
             receiver (Mechanism object)
             mappingFunction (Function of type kwMappingFunction)
        :return (list):
        """
        return self.configuration

    def get_mechanism_dict(self):
        """Return mechanismDict (dict of mechanisms in configuration)
        The key of each config_entry is the name of a mechanism, and the value the corresponding Mechanism object
        :return (dict):
        """
        return self.mechanismDict

    @property
    def variableInstanceDefault(self):
        return self._variableInstanceDefault

    @variableInstanceDefault.setter
    def variableInstanceDefault(self, value):
        assigned = -1
        try:
            value
        except ValueError as e:
            pass
        self._variableInstanceDefault = value
