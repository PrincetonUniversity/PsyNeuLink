#
# *********************************************  Process ***************************************************************
#

import re
import Functions
from Functions.ShellClasses import *
from Globals.Registry import register_category
from Functions.Mechanisms.Mechanism import Mechanism_Base
from Functions.Mechanisms.Mechanism import mechanism
from toposort import *

# *****************************************    SYSTEM CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# Labels for items in configuration entry tuples
PROCESS = 0
INPUT = 1

SystemRegistry = {}

kwSystemInputState = 'SystemInputState'
from Functions.MechanismStates.MechanismOutputState import MechanismOutputState


# class SystemInputState(MechanismOutputState):
#     """Represent input to System and provide to first Mechanism of each Process in the Configuration
#
#     Each instance encodes an item of the System input (one of the 1D arrays in the 2D np.array input) and provides that
#         input to a Mapping projection to an inputState of the 1st Mechanism of each Process in the Configuration;
#         see Process Description for mapping when there is more than one Process input value and/or Mechanism inputState
#
#      Notes:
#       * Declared as sublcass of MechanismOutputState so that it is recognized as a legitimate sender to a Projection
#            in Projection.instantiate_sender()
#       * self.value is used to represent input to Process provided as variable arg on command line
#
#     """
#     def __init__(self, owner=None, variable=NotImplemented, prefs=NotImplemented):
#         """Pass variable to mapping projection from System to first Mechanism of each Process in Configuration
#
#         :param variable:
#         """
#         self.name = owner.name + "_" + kwSystemInputState
#         self.prefs = prefs
#         self.sendsToProjections = []
#         self.ownerMechanism = owner
#         self.value = variable


class SystemError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)

# FIX:
# DOCUMENT:  EDIT ALL REFERENCES TO CONGIFURATION -> Process
# DOCUMENT:  Self.processes must be a list of Processes that will compiled into a graph of mechanisms for execution
# FIX:  NEED TO CREATE THE PROJECTIONS FROM THE PROCESS TO THE FIRST MECHANISM IN PROCESS FIRST SINCE,
# FIX:  ONCE IT IS IN THE GRAPH, IT IS NOT LONGER EASY TO DETERMINE WHICH IS WHICH IS WHICH (SINCE SETS ARE NOT ORDERED)
class System_Base(System):
    """Implement abstract class for System category of Function class

    Description:
        A System is defined by a kwProcesses param (list of Processes) and a time scale.  Executing a System executes
            the Mechanisms in the Processes in a topologically sorted order, based on their sequence in the Processes.

    Instantiation:
        A System can be instantiated in one of two ways:
            - by calling <System>()
        A System is instantiated by assigning:
            - the Mechanisms in all of the Processes in kwProcesses to a graph that is topologically sorted into
                 a sequentially ordered list of sets containing mechanisms to be executed at the same time
            - each input in it's input list to the first Mechanism of each Process
            - the outputs of the Mechanisms in the last set of the graph to System.outputState(s)

    Initialization arguments:
        - input (list of values): one item for each Mechanism in the 1st set of graph
            (default: variableInstanceDefault for the first Mechanism in each Process)
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
    functionType = "System"

    registry = SystemRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    # Use inputValueSystemDefault as default input to process
    variableClassDefault = inputValueSystemDefault

    # FIX: default Process
    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({kwProcesses: [],
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
        self.processes = []
        self.mechanismDict = {}
        register_category(self, System_Base, SystemRegistry, context=context)

        if context is NotImplemented:
            # context = self.__class__.__name__
            context = kwInit + self.name

        super(System_Base, self).__init__(variable_default=default_input_value,
                                           param_defaults=params,
                                           name=self.name,
                                           prefs=prefs,
                                           context=context)
        # IMPLEMENT CORRECT REPORTING HERE
        # if self.prefs.reportOutputPref:
        #     print("\n{0} initialized with:\n- configuration: [{1}]".
        #           # format(self.name, self.configurationMechanismNames.__str__().strip("[]")))
        #           format(self.name, self.mechanism_names.__str__().strip("[]")))

    def validate_variable(self, variable, context=NotImplemented):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        :param variable:
        :param context:
        :return:
        """

        super(System_Base, self).validate_variable(variable, context)

        # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Call methods that must be run before execute method is instantiated

        Need to do this before instantiate_execute_method as mechanisms in configuration must be instantiated
            in order to assign input projection and self.outputState to first and last mechanisms, respectively

        :param context:
        :return:
        """
        self.instantiate_graph(inputs=self.variable, context=context)

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
            print("System object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[kwExecuteMethod], kwExecuteMethod)
            self.paramsCurrent[kwExecuteMethod] = self.execute
        # If validation pref is set, instantiate and execute the System
        if self.prefs.paramValidationPref:
            super(System_Base, self).instantiate_execute_method(context=context)
        # Otherwise, just set System output info to the corresponding info for the last mechanism(s) in the Process(es)
        else:
            self.value = self.configuration[-1][OBJECT].outputState.value

# DOCUMENTATION:
#         Uses paramClassDefaults[kwConfiguration] == [Mechanism_Base.defaultMechanism] as default
#
#         1) Iterate through Process.mechanism_list for each Process in self.processes
#               - Create Subsequent: {Previous} for each sequential pair in the Configuration
#               - Add each pair to self.graph
#         2) Call toposort_flatten(self.graph) to get a list of sets of mechanisms to be executed in order
#
# FIX:
#     ** PROBLEM: self.value IS ASSIGNED TO variableInstanceDefault WHICH IS 2D ARRAY,
        # BUT PROJECTION EXECUTION FUNCTION TAKES 1D ARRAY
#         Assign projection from Process (self.value) to inputState of the first mechanism in the configuration
#     **?? WHY DO THIS, IF SELF.VALUE HAS BEEN ASSIGNED AN INPUT VALUE, AND PROJECTION IS PROVIDING INPUT TO MECHANISM??
#         Assigns variableInstanceDefault to variableInstanceDefault of first mechanism in configuration

# FIX: ALLOW Projections (??ProjectionTiming TUPLES) TO BE INTERPOSED BETWEEN MECHANISMS IN CONFIGURATION
# FIX: AUGMENT LinearMatrix TO USE kwFullConnectivityMatrix IF len(sender) != len(receiver)

    def instantiate_graph(self, inputs=None, context=NotImplemented):
        """Create topologically sorted graph of Mechanisms from Processes and use to execute them in hierarchical order

        Iterate through self.processes, instantiating each (including the input to each input projection)
        Iterate through Process.mechanism_list for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.graph
        Call toposort_flatten(self.graph) to generate a sequential list of Mechanisms to be executed in order

        :param context:
        :return:
        """

        processes = self.paramsCurrent[kwProcesses]
        self.graph = {}
        self.mechanisms = {}

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS
        for i in range(len(processes)):

            # Convert all entries to (process, input) tuples, with None as filler for absent inputs
            if not isinstance(processes[i], tuple):
                processes[i] = (processes[i], None)

            # If input was provided on command line, assign that to input item of tuple
            if inputs:
                # Number of inputs in variable must equal number of Processes
                if len(inputs) != len(processes):
                    raise SystemError("Number of inputs ({0}_must equal number of Processes in kwProcesses ({1})".
                                      format(len(inputs), len(processes)))
                # Replace input item in tuple with one from variable
                processes[i] = (processes[i][PROCESS], inputs[i])
            # Validate input
            if processes[i][INPUT] and not isinstance(processes[i][INPUT], (numbers.Number, list, np.ndarray)):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, processes[i][INPUT]))

            process = processes[i][PROCESS]
            input = processes[i][INPUT]

            # If process item is a Process object, assign input as default
            if isinstance(process, Process):
                if input:
                    process.assign_defaults(input)

            # Otherwise, instantiate Process
            if not isinstance(process, Process):
                if inspect.isclass(process) and issubclass(process, Process):
                    # FIX: MAKE SURE THIS IS CORRECT
                    # Provide self as context, so that Process knows it is part of a Sysetm (and which one)
                    # Note: this is used by Process.instantiate_configuration() when instantiating first Mechanism
                    #           in Configuration, to override instantiation of projections from Process.input_state
                    process = Process(default_input_value=input, context=self)
                elif isinstance(process, dict):
                    # IMPLEMENT:  HANDLE Process specification dict here; include input as ??param, and context=self
                    raise SystemError("Attempt to instantiate process {0} in kwProcesses of {1} "
                                      "using a Process specification dict: not currently supported".
                                      format(process.name, self.name))
                else:
                    raise SystemError("Entry {0} of kwProcesses ({1}) must be a Process object, class, or a"
                                      "specification dict , ".format(i, process))

            # FIX: SHOULD BE ABLE TO PASS INPUTS HERE, NO?  PASSED IN VIA VARIABLE, ONE FOR EACH PROCESS
            # FIX: MODIFY instantiate_configuration TO ACCEPT input AS ARG
            # NEEDED?? WASN"T IT INSTANTIATED ABOVE WHEN PROCESS WAS INSTANTIATED??
            # process.instantiate_configuration(self.variable[i], context=context)

            # Iterate through all but last Mechanism in Process' mechanism_list to:
            # - assign receiver:sender pairs to graph dict
            # - assign sender mechanism entry in self.mechanisms dict, with mech as key and its Process as value
            for j in range(len(process.mechanism_list)-1):
                sender_mech_tuple = process.mechanism_list[j]
                receiver_mech_tuple = process.mechanism_list[j+1]
                self.graph[receiver_mech_tuple] = {sender_mech_tuple}
                self.mechanisms[sender_mech_tuple[0]] = process.name

            # Create toposort tree and instance of sequential list:
            self.execution_sets = toposort(self.graph)
            self.execution_list = toposort_flatten(self.graph)

            # FIX: ASSIGN THIRD ITEM OF EACH mech_tuple TO BE SET IN WHICH MECH IS NOW PLACED (BY TOPOSORT)

            print (self.execution_sets)
            print (self.execution_list)
        #endregion
        temp = True


# FIX: MAY NEED TO ASSIGN OWNERSHIP OF MECHANISMS IN PROCESSES TO THEIR PROCESSES (OR AT LEAST THE FIRST ONE)
# FIX: SO THAT INPUT CAN BE ASSIGNED TO CORRECT FIRST MECHANISMS (SET IN GRAPH DOES NOT KEEP TRACK OF ORDER)
# FIX: ENTRIES IN GRAPH SHOULD BE 3-ITEM TUPLES, WITH THIRD THE SET (IN TOPOSORT SEQUENCE) TO WHICH EACH ITEM BELONGS
    def execute(self,
                inputs=None,
                time_scale=NotImplemented,
                context=NotImplemented
                ):
        """Coordinate execution of mechanisms in project list (self.configuration)

        Assign items in input to corresponding Processes (in self.params[kwProcesses])
        Go through mechanisms in execution_list, and execute each one in the order they appear in the list

        ** MORE DOCUMENTATION HERE

        Arguments:
# DOCUMENT:
        - time_scale (TimeScale enum): determines whether mechanisms are executed for a single time step or a trial
        - context (str): not currently used

        Returns: System.outputState

        :param input:  (list of values)
        :param time_scale:  (TimeScale) - (default: TRIAL)
        :param context: (str)
        :return: (value)
        """

        if context is NotImplemented:
            context = kwExecuting + self.name

        #region ASSIGN INPUTS TO PROCESSES
        # Assign each item in input to corresponding Process;
        #    it will be assigned as the value of Process.input_state which, in turn, will be used as
        #    the input to the mapping projection to the first Mechanism in that Process' configuration
        if inputs:
            if len(inputs) != len(self.processes):
                raise SystemError("Number of inputs ({0}) must match number of processes in kwProcesses ({1})".
                                  format(len(inputs), len(self.processes)))
            for i in range(len(inputs)):
                input = input[i]
                process = self.processes[i][PROCESS]

                # Make sure there is an input, and if so convert it to 2D np.ndarray (required by Process
                if not input or input is NotImplemented:
                    continue
                else:
                    # Assign input as value of corresponding Process inputState
                    process.assign_input_values(input=input, context=context)
        #endregion

        #region EXECUTE EACH MECHANISM
        # Execute each Mechanism in self.execution_list, in the order listed
        for i in range(len(self.execution_list)):

            # FIX: NEED TO DEAL WITH CLOCK HERE (SHOULD ONLY UPDATE AFTER EACH SET IN self.exuection_sets
            # FIX: SET TO THIRD ITEM IN MECHANISM TUPLE, WHICH INDICATES THE TOPOSORT SET
            CentralClock.time_step = i
            mechanism, params = self.execution_list[i]
            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
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
