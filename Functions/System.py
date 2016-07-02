#
# *********************************************  Process ***************************************************************
#

import re
import math
from collections import UserList
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

MECHANISM = 0
PHASE_SPEC = 1
PARAMS = 2

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


class TerminalMechanismList(UserList):
    """Return Mechanism item from (Mechanism, runtime_params) tuple in self.terminal_mech_tuples

    Convenience class, that returns Mechanism item of indexed tuple in self.terminal_mech_tuples

    """
    def __init__(self, system):
        super(TerminalMechanismList, self).__init__()
        self.mech_tuples = system.terminal_mech_tuples

    def __getitem__(self, item):
        return self.mech_tuples[item][0]

    def __setitem__(self, key, value):
        raise ("MyList is read only ")

# FIX:  NEED TO CREATE THE PROJECTIONS FROM THE PROCESS TO THE FIRST MECHANISM IN PROCESS FIRST SINCE,
# FIX:  ONCE IT IS IN THE GRAPH, IT IS NOT LONGER EASY TO DETERMINE WHICH IS WHICH IS WHICH (SINCE SETS ARE NOT ORDERED)
class System_Base(System):
    """Implement abstract class for System category of Function class

    Description:
        A System is defined by a kwProcesses param (list of Processes) and a time scale.  Executing a System executes
            the Mechanisms in the Processes in a topologically sorted order, based on their sequence in the Processes.

    Instantiation:
        A System can be instantiated in one of two ways:
            [TBI: - Calling the run() function, which instantiates a default System]
            - by calling System(<args>)
        A System is instantiated by assigning:
            - the Mechanisms in all of the Processes in kwProcesses to a graph that is topologically sorted into
                 a sequentially ordered list of sets containing mechanisms to be executed at the same time
            - each input in it's input list to the first Mechanism of each Process
            - the outputs of terminal Mechanisms in the graph System.outputState(s)
                (terminal mechanisms are ones that do not project to any other mechanisms in the System)

    Initialization arguments:
        - input (list of values): list of inputs (2D np.arrays), one for each Process in kwProcesses
            [??CORRECT: one item for each originMechanism (Mechanisms in the 1st set of self.graph)]
            (default: variableInstanceDefault for the first Mechanism in each Process)
        - params (dict):
            kwProcesses (list): (default: a single instance of the default Process)
            kwController (list): (default: SystemDefaultController)
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): used to track object/class assignments and methods in hierarchy

        NOTES:
            * if kwProcesses or time_scale are not provided:
                a single default Process is instantiated and TimeScale.TRIAL are used

    SystemRegistry:
        All Processes are registered in ProcessRegistry, which maintains a dict for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Processes can be named explicitly (using the name='<name>' argument).  If this argument is omitted,
        it will be assigned "Mapping" with a hyphenated, indexed suffix ('Mapping-n')

# DOCUMENTATION: UPDATE Execution BELOW
    Execution:
        - System.execute calls mechanism.update_states_and_execute for each mechanism in its configuration in sequence
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
        + paramClassDefaults = {kwProcesses: [Mechanism_Base.defaultMechanism],
                                kwController: SystemDefaultController,
                                kwTimeScale: TimeScale.TRIAL}


    Class methods:
        • validate_variable(variable, context):  insures that variable is 3D np.array (one 2D for each Process)
        • instantiate_attributes_before_execute_method(context):  calls self.instantiate_graph
        • instantiate_execute_method(context): validates only if self.prefs.paramValidationPref is set
        • instantiate_graph(inputs, context):  instantiates Processes in self.process and constructs execution_list
        • assign_output_states(): identifies terminal Mechanisms in self.graph and assigns self.outputStates to them
        • execute(inputs, time_scale, context):  executes Mechanisms in order specified by execution_list
        • variableInstanceDefaults(value):  setter for variableInstanceDefaults;  does some kind of error checking??

    Instance attributes:
        + processes (list of (Process, input) tuples):  an ordered list of Processes and corresponding inputs;
            derived from self.inputs and params[kwProcesses], and used to construct self.graph and execute the System
                 (default: a single instance of the default Process)
        + phaseSpecMax (int) - maximum phaseSpec for all Mechanisms in System
    [TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each Process in self.processes
        + output (list): containts Process.ouput for each Process in self.processes
    ]

        [TBI: + inputs (list): each item is the Process.input object for the corresponding Process in self.processes]
        [TBI: + outputs (list): each item is the Process.output object for the corresponding Process in self.processes]
        + graph (dict): each entry is <Receiver>: {sender, sender...} pairing
        + execution_sets (list of sets):
            each set contains mechanism to be executed at the same time;
            the sets are ordered in the sequence with which they should be executed
        + execute_list (list of Mechanisms):  a list of Mechanisms in the order they should be executed;
            Note: the list is a random sample subject to the constraints of ordering in self.execute_sets
        [TBI: + originMechanisms (list):  Mechanism objects without projections from any other Mechanisms in the System]
        + mechanisms (dict): dict of Mechanism:Process entries for all Mechanisms in the System
            the key for each entry is a Mechanism object
            the value of each entry is a list of Process.name strings (as a mechanism can be in several Processes)
        + terminalMechanisms (list):  Mechanism objects without projections to any other Mechanisms in the System
            Notes:
            * this is a convenience, read-only list
            * each item refers to the Mechanism item of the corresponding tuple in self.terminal_mech_tuples
            * the Mechanisms in this list provide the output values for System.output
        + terminal_mech_tuples (list):  Mechanisms without projections to any other Mechanisms in the System
            Notes:
            * each item is a (Mechanism, runtime_params) tuple
            * each tuple is an entry of Process.mechanism_list for the Process in which the Mechanism occurs
            * each tuple serves as the key for the mechanism in self.graph
        [TBI: + controller (SystemControlMechanism): the control mechanism assigned to the System]
            (default: SystemDefaultController)
        + value (3D np.array):  each 2D array item the value (output) of the corresponding Process in kwProcesses
        + timeScale (TimeScale): set in params[kwTimeScale]
             defines the temporal "granularity" of the process; must be of type TimeScale
                (default: TimeScale.TRIAL)
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
    from Functions import SystemDefaultController
    from Functions import DefaultController
    from Functions import Goofiness
    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({kwProcesses: [],
                               kwController: DefaultController,
                               kwControllerPhaseSpec: 0,
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
        self.outputStates = {}
        self.phaseSpecMax = 0

        register_category(self, System_Base, SystemRegistry, context=context)

        if context is NotImplemented:
            # context = self.__class__.__name__
            context = kwInit + self.name

        super(System_Base, self).__init__(variable_default=default_input_value,
                                           param_defaults=params,
                                           name=self.name,
                                           prefs=prefs,
                                           context=context)

        self.controller = self.paramsCurrent[kwController](params={kwSystem: self})
        try:
            # Get controller phaseSpec
            self.phaseSpecMax = max(self.phaseSpecMax, self.controller.phaseSpec)
        except (AttributeError, TypeError):
            # Controller phaseSpec not specified
            try:
                # Assign System specification of Controller phaseSpec if provided
                self.controller.phaseSpec = self.paramsCurrent[kwControllerPhaseSpec]
                self.phaseSpecMax = max(self.phaseSpecMax, self.controller.phaseSpec)
            except:
                # No System specification, so use System max as default
                self.controller.phaseSpec = self.phaseSpecMax

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

        # MODIFIED 6/26/16 OLD:
        # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)
#
# FIX:  THIS CURRENTLY FAILS:
        # # MODIFIED 6/26/16 NEW:
        # # Force System variable specification to be a 3D array (to accommodate input states for each Process):
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 3)
        # self.variable = convert_to_np_array(self.variable, 3)

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Call instantiate_graph and assign self.controller

        These must be done before instantiate_execute_method as the latter may be called during init for validation
        """

        self.instantiate_graph(inputs=self.variable, context=context)

        # MODIFIED 6/28/16: OLD
        # self.controller = self.paramsCurrent[kwController]
        # MODIFIED 6/28/16: NEW - MOVED TO self.__init__()

    def instantiate_execute_method(self, context=NotImplemented):
        """Override Function.instantiate_execute_method:

        This is necessary to:
        - insure there is no kwExecuteMethod specified (not allowed for a System object)
        - suppress validation (and attendant execution) of System execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in kwProcesses have already been validated

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
        # Otherwise, just set System output info to the corresponding info for the last mechanism(s) in self.processes
        else:
            self.value = self.processes[-1][PROCESS].outputState.value

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

        If self.processes is empty, instantiate default Process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        Iterate through Process.mechanism_list for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.graph
        Call toposort_flatten(self.graph) to generate a sequential list of Mechanisms to be executed in order

        :param context:
        :return:
        """

        self.processes = self.paramsCurrent[kwProcesses]
        self.graph = {}
        self.mechanisms = {}

        # Assign default Process if kwProcess is empty, or invalid
        if not self.processes:
            from Functions.Process import Process_Base
            self.processes.append((Process_Base(), None))

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS
        for i in range(len(self.processes)):

            # Convert all entries to (process, input) tuples, with None as filler for absent inputs
            if not isinstance(self.processes[i], tuple):
                self.processes[i] = (self.processes[i], None)

            # If input was provided on command line, assign that to input item of tuple
            if inputs:
                # Number of inputs in variable must equal number of self.processes
                if len(inputs) != len(self.processes):
                    raise SystemError("Number of inputs ({0}_must equal number of Processes in kwProcesses ({1})".
                                      format(len(inputs), len(self.processes)))
                # Replace input item in tuple with one from variable
                self.processes[i] = (self.processes[i][PROCESS], inputs[i])
            # Validate input
            if self.processes[i][INPUT] and not isinstance(self.processes[i][INPUT],(numbers.Number, list, np.ndarray)):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, self.processes[i][INPUT]))

            process = self.processes[i][PROCESS]
            input = self.processes[i][INPUT]

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
                    raise SystemError("Entry {0} of kwProcesses ({1}) must be a Process object, class, or a "
                                      "specification dict for a Process".format(i, process))


            # process should be a Process object

            # Assign the Process a reference to this System
            process.system = self

            # Get max of Process phaseSpecs
            self.phaseSpecMax = int(max(math.floor(process.phaseSpecMax), self.phaseSpecMax))

            # FIX: SHOULD BE ABLE TO PASS INPUTS HERE, NO?  PASSED IN VIA VARIABLE, ONE FOR EACH PROCESS
            # FIX: MODIFY instantiate_configuration TO ACCEPT input AS ARG
            # NEEDED?? WASN"T IT INSTANTIATED ABOVE WHEN PROCESS WAS INSTANTIATED??
            # process.instantiate_configuration(self.variable[i], context=context)

            # Iterate through mechanism tuples in Process' mechanism_list
            for j in range(len(process.mechanism_list)-1):

                sender_mech_tuple = process.mechanism_list[j]
                receiver_mech_tuple = process.mechanism_list[j+1]

                # For first Mechanism in list, if sender has a projection from Process.input_state, treat as "root"
                if j==0:
                    if sender_mech_tuple[MECHANISM].receivesProcessInput:
                        self.graph[sender_mech_tuple] = set()

                # For all others in list:
                # - assign receiver-sender pair as entry self.graph dict
                # - assign sender mechanism entry in self.mechanisms dict, with mech as key and its Process as value
                #     (this is used by Process.instantiate_configuration() to determine if Process is part of System)
                if receiver_mech_tuple in self.graph:
                    # If the receiver is already in the graph, add the sender to its sender set
                    self.graph[receiver_mech_tuple].add(sender_mech_tuple)
                else:
                    # If the receiver is NOT already in the graph, assign the sender in a set
                    self.graph[receiver_mech_tuple] = {sender_mech_tuple}

                # If the sender is already in the System's mechanisms dict
                if sender_mech_tuple[MECHANISM] in self.mechanisms:
                    # Add to entry's list
                    self.mechanisms[sender_mech_tuple[MECHANISM]].append(process.name)
                else:
                    # Add new entry
                    self.mechanisms[sender_mech_tuple[MECHANISM]] = [process.name]

        # Create toposort tree and instance of sequential list:
        self.execution_sets = list(toposort(self.graph))
        self.execution_list = toposort_flatten(self.graph, sort=False)

        self.assign_output_states()

        # FIX: ASSIGN THIRD ITEM OF EACH mech_tuple TO BE SET IN WHICH MECH IS NOW PLACED (BY TOPOSORT)

        print (self.execution_sets)
        print (self.execution_list)
        #endregion
        temp = True


# FIX: MAY NEED TO ASSIGN OWNERSHIP OF MECHANISMS IN PROCESSES TO THEIR PROCESSES (OR AT LEAST THE FIRST ONE)
# FIX: SO THAT INPUT CAN BE ASSIGNED TO CORRECT FIRST MECHANISMS (SET IN GRAPH DOES NOT KEEP TRACK OF ORDER)
# FIX: ENTRIES IN GRAPH SHOULD BE 3-ITEM TUPLES, WITH THIRD THE SET (IN TOPOSORT SEQUENCE) TO WHICH EACH ITEM BELONGS
    def assign_output_states(self):
        """Find terminal Mechanisms of graph, and use to assign self.output_states and self.terminalMechanisms

        Identifies terminal Mechanisms in graph (ones that do not have have projections to any other mechanisms)
        Assigns the (Mechanism, runtime_params) tuple for each in Process to self.terminal_mech_tuples
        Instantiates self.terminalMechanisms:
            this is a convenience list that refers to the Mechanism item of each tuple in self.terminal_mech_tuples
        Assigns the outputState for each Mechanism in self.terminalMechanisms to self.outputStates
        """
        receiver_mech_tuples = set(list(self.graph.keys()))
        sender_mech_tuples = set()
        for receiver in self.graph:
            sender_mech_tuples = sender_mech_tuples.union(self.graph[receiver])
        self.terminal_mech_tuples = list(receiver_mech_tuples - sender_mech_tuples)
        self.terminalMechanisms = TerminalMechanismList(self)
        for mech in self.terminalMechanisms:
            self.outputStates[mech.name] = mech.outputStates



    def execute(self,
                inputs=None,
                time_scale=NotImplemented,
                context=NotImplemented
                ):
# DOCUMENT: NEEDED — INCLUDED HANDLING OF phaseSpec
        """Coordinate execution of mechanisms in process list (self.processes)

        Assign items in input to corresponding Processes (in self.params[kwProcesses])
        Go through mechanisms in execution_list, and execute each one in the order they appear in the list

        ** MORE DOCUMENTATION HERE

        Arguments:
# DOCUMENT:
        - time_scale (TimeScale enum): determines whether mechanisms are executed for a single time step or a trial
        - context (str): not currently used

        :param input:  (list of values)
        :param time_scale:  (TimeScale) - (default: TRIAL)
        :param context: (str)
        :return: (value)
        """

        if context is NotImplemented:
            context = kwExecuting + self.name

        if time_scale is NotImplemented:
            self.timeScale = TimeScale.TRIAL

        #region ASSIGN INPUTS TO PROCESSES
        # Assign each item in input to corresponding Process;
        #    it will be assigned as the value of Process.input_state which, in turn, will be used as
        #    the input to the mapping projection to the first Mechanism in that Process' configuration
        if inputs:
            if len(inputs) != len(self.processes):
                raise SystemError("Number of inputs ({0}) to {1} does not match its number of processes ({2})".
                                  format(len(inputs), self.name,  len(self.processes) ))
            for i in range(len(inputs)):
                input = inputs[i]
                process = self.processes[i][PROCESS]

                # Make sure there is an input, and if so convert it to 2D np.ndarray (required by Process
                if not input or input is NotImplemented:
                    continue
                else:
                    # Assign input as value of corresponding Process inputState
                    process.assign_input_values(input=input, context=context)
        #endregion


        #region EXECUTE CONTROLLER
        try:
            if self.controller.phaseSpec == (CentralClock.time_step % (self.phaseSpecMax +1)):
                self.controller.update(time_scale=TimeScale.TRIAL,
                                       runtime_params=NotImplemented,
                                       context=NotImplemented)
        except AttributeError:
            if not 'INIT' in context:
                raise SystemError("PROGRAM ERROR: no controller instantiated for {0}".format(self.name))
        #endregion

        #region EXECUTE EACH MECHANISM

        # Print output value of primary (first) outpstate of each terminal Mechanism in System
        if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
            print("\n{0} BEGUN EXECUTION (time_step {1}) **********".format(self.name, CentralClock.time_step))

        # Execute each Mechanism in self.execution_list, in the order listed
        for i in range(len(self.execution_list)):

            mechanism, params, phase_spec = self.execution_list[i]

            # Only update Mechanism on time_step(s) determined by its phaseSpec (specified in Mechanism's Process entry)
# FIX: NEED TO IMPLEMENT FRACTIONAL UPDATES (IN Mechanism.update()) FOR phaseSpec VALUES THAT HAVE A DECIMAL COMPONENT
            if phase_spec == (CentralClock.time_step % (self.phaseSpecMax +1)):
                # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
                mechanism.update(time_scale=self.timeScale,
                                 runtime_params=params,
                                 context=context)
                # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR MECHANISM ITSELF
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

        # if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
        #     print("\n{0} completed:\n- output: {1}".format(self.name,
        #                                                    re.sub('[\[,\],\n]','',str(self.outputState.value))))

        # Print output value of primary (first) outpstate of each terminal Mechanism in System
        if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
            print("\n{0} COMPLETED (time_step {1}) *******".format(self.name, CentralClock.time_step))
            for mech in self.terminal_mech_tuples:
                if mech[MECHANISM].phaseSpec == (CentralClock.time_step % (self.phaseSpecMax + 1)):
                    print("- output for {0}: {1}".format(mech[MECHANISM].name,
                                                         re.sub('[\[,\],\n]','',str(mech[MECHANISM].outputState.value))))


    @property
    def variableInstanceDefault(self):
        return self._variableInstanceDefault

    @variableInstanceDefault.setter
    def variableInstanceDefault(self, value):
# FIX: WHAT IS GOING ON HERE?  WAS THIS FOR DEBUGGING?  REMOVE??
        assigned = -1
        try:
            value
        except ValueError as e:
            pass
        self._variableInstanceDefault = value
