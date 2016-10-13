# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  Process ***************************************************************
#

import math
import re
from collections import UserList, Iterable

from toposort import *
# from PsyNeuLink.Globals.toposort.toposort import toposort

from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Functions.ShellClasses import *
from PsyNeuLink.Functions.Mechanisms.Mechanism import MonitoredOutputStatesOption
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import Comparator
from PsyNeuLink.Functions.Process import ProcessInputState
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base

# *****************************************    SYSTEM CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# Labels for items in configuration tuples
PROCESS = 0
PROCESS_INPUT = 1

MECHANISM = 0
PARAMS = 1
PHASE_SPEC = 2

SystemRegistry = {}

kwSystemInputState = 'SystemInputState'


class SystemError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


class ProcessList(UserList):
    """Provides access to items from (process, process_input) tuples in a list of process tuples

    Process tuples must be of the following form:  (process object, process_input list or array)

    """
    def __init__(self, system):
        super(ProcessList, self).__init__()
        try:
            self.process_tuples
        except AttributeError:
            raise SystemError("{0} subclass of ProcessList must assign process_tuples attribute".
                              format({self.__class__}))

    def __getitem__(self, item):
        # return self.mech_tuples[item][0]
        # return next(iter(self.mech_tuples[item]))
        return list(self.process_tuples[item])[PROCESS]
        # return list(self.mech_tuples[item])

    def __setitem__(self, key, value):
        raise ("MyList is read only ")

    def __len__(self):
        return (len(self.process_tuples))

    @property
    def processes(self):
        return list(item[PROCESS] for item in self.process_tuples)

    @property
    def processNames(self):
        return list(item[PROCESS].name for item in self.process_tuples)


class SystemProcessList(ProcessList):
    """Provide access to lists of mechanisms and their attributes from tuples list in <process>.mechanismList
    """
    def __init__(self, system):
        self.process_tuples = system.process_tuples
        super().__init__(system)

    def get_tuple_for_process(self, process):
        """Return mechanism tuple containing specified mechanism from <process>.mechanismList
        """
        # PROBLEM: IF PROCESS APPEARS IN MORE THAN ONE TUPLE, WILL ONLY RETURN THE FIRST
        return next((process_tuple for process_tuple in self.process_tuples if process_tuple[PROCESS] is process), None)


class MechanismList(UserList):
    """Provides access to items from (Mechanism, runtime_params, phase) tuples in a list of mechanism tuples

    Mechanism tuples must be of the following form:  (mechanism object, runtime_params dict, phaseSpec int)

    """
    def __init__(self, system):
        super(MechanismList, self).__init__()
        try:
            self.mech_tuples
        except AttributeError:
            raise SystemError("{0} subclass of MechanismList must assign mech_tuples attribute".
                              format({self.__class__}))

    def __getitem__(self, item):
        # return self.mech_tuples[item][0]
        # return next(iter(self.mech_tuples[item]))
        return list(self.mech_tuples[item])[MECHANISM]
        # return list(self.mech_tuples[item])

    def __setitem__(self, key, value):
        raise ("MyList is read only ")

    def __len__(self):
        return (len(self.mech_tuples))

    @property
    def mechanisms(self):
        return list(self)

    @property
    def mechanismNames(self):
        # names = []
        # for item in self.mechanisms:
        #     names.append(item.name)
        # return names
        return list(item.name for item in self.mechanisms)

    @property
    def mechanismValues(self):
        # values = []
        # for item in self.mechanisms:
        #     values.append(item.value)
        # return values
        return list(item[MECHANISM].value for item in self.mechanisms)

    @property
    def outputStateNames(self):
        names = []
        for item in self.mechanisms:
            for output_state in item.outputStates:
                names.append(output_state)
        return names

    @property
    def outputStateValues(self):
        """Return values of outputStates for all mechanisms in list
        """
        values = []
        for item in self.mechanisms:
            for output_state_name, output_state in list(item.outputStates.items()):
                # output_state_value = output_state.value
                # if isinstance(output_state_value, Iterable):
                #     output_state_value = list(output_state_value)
                # values.append(output_state_value)
                # # MODIFIED 9/15/16 OLD:
                # values.append(float(output_state.value))
                # MODIFIED 9/15/16 NEW:
                values.append(output_state.value)
                # MODIFIED 9/15/16 END
        return values


class ProcessMechanismsList(MechanismList):
    """Provide access to lists of mechanisms and their attributes from tuples list in <process>.mechanismList
    """
    def __init__(self, process):
        self.mech_tuples = process.mechanismList
        super().__init__(system)

    def get_tuple_for_mech(self, mech):
        """Return mechanism tuple containing specified mechanism from <process>.mechanismList
        """
        # PROBLEM: IF MECHANISM APPEARS IN MORE THAN ONE TUPLE, WILL ONLY RETURN THE FIRST
        return next((mech_tuple for mech_tuple in self.mech_tuples if mech_tuple[0] is mech), None)


class SystemMechanismsList(MechanismList):
    """Provide access to lists of mechanisms and their attributes from tuples list in <system>.mechanismList
    """
    def __init__(self, system, tuples_list):
        self.mech_tuples = tuples_list
        super().__init__(system)

    def get_tuple_for_mech(self, mech):
        """Return mechanism tuple containing specified mechanism from <process>.mechanismList
        """
        # PROBLEM: IF MECHANISM APPEARS IN MORE THAN ONE TUPLE, WILL ONLY RETURN THE FIRST
        return next((mech_tuple for mech_tuple in self.mech_tuples if mech_tuple[0] is mech), None)


class OriginMechanismsList(MechanismList):
    """Provide access to lists of origin mechanisms and their attributes from tuples list in <system>.origin_mech_tuples
    """
    def __init__(self, system):
        self.mech_tuples = system.origin_mech_tuples
        super().__init__(system)


class TerminalMechanismsList(MechanismList):
    """Provide access to lists of terminal mechs and their attribs from tuples list in <system>.terminal_mech_tuples
    """
    def __init__(self, system):
        self.mech_tuples = system.terminal_mech_tuples
        super().__init__(system)


# FIX:  IMPLEMENT DEFAULT PROCESS
# FIX:  NEED TO CREATE THE PROJECTIONS FROM THE PROCESS TO THE FIRST MECHANISM IN PROCESS FIRST SINCE,
# FIX:  ONCE IT IS IN THE GRAPH, IT IS NOT LONGER EASY TO DETERMINE WHICH IS WHICH IS WHICH (SINCE SETS ARE NOT ORDERED)

from PsyNeuLink.Functions import SystemDefaultControlMechanism
from PsyNeuLink.Functions.Process import process


# System factory method:
@tc.typecheck
def system(default_input_value=None,
           processes:list=[],
           initial_values:dict={},
           controller=SystemDefaultControlMechanism,
           enable_controller:bool=False,
           monitored_output_states:list=[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES],
           params:tc.optional(dict)=None,
           name:tc.optional(str)=None,
           prefs:is_pref_set=None,
           context=None):

    """Return instance of System_Base

    If called with no arguments, return an instance of system with a single default process and mechanism
    If called with a name string, use it as the name of the system instance returned
    If a params dictionary is included, it is passed to the system

    See System_Base for class description
    """

    # Called with descriptor keyword
    if not processes:
        processes = [process()]

    return System_Base(default_input_value=default_input_value,
                       processes=processes,
                       controller=controller,
                       initial_values=initial_values,
                       enable_controller=enable_controller,
                       monitored_output_states=monitored_output_states,
                       params=params,
                       name=name,
                       prefs=prefs,
                       context=context)


class System_Base(System):
    # DOCUMENT: enable_controller option
    # DOCUMENT: ALLOWABLE FEEDBACK CONNECTIONS AND HOW THEY ARE HANDLED
    """Abstract class for System

        Should be instantiated using the system() factory method

        A system is a collection of processes that are run together.  Running a system executes all of the mechanisms
            in its processes.  Mechanisms are executed in a topologically sorted order, based on the order in which
            they are listed in their processes and any projections among them.  Projections between processes
            are permitted, as are recurrent projections.

        Mechanisms within a system are designated as:
            ORIGIN: receives input to the system, and begins execution
            TERMINAL: final point of execution, and provides an output of the system
            SINGLETON: both an ORIGIN and a TERMINAL
            INITIATE_CYCLE: closes a recurrent loop, and can receive an initial value specification
            CYCLE: receives a projection that closes a recurrent loop
            MONITORING: monitors value of another mechanism for use in learning
            CONTROL:  monitors value of another mechanism for use in real-time control
            INTERNAL: processing mechanism that does not fall into any of the categories above

        Systems are represented in a graph structure (see graph),
        that can be passed for analysis to graph theoretical tools


    Parameters
    ----------
    input : list of 2d ndarrays : default variableInstanceDefault for the first mechanism in each process
        One array for each process in the system

    processes : list of process objects : default a single instance of the ??DefaultProcess??

    controller :

    initial_values :

    enable_controller :

    monitored_output_stated :

    params :

    name :

    prefs :

    context :

    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    Initialization arguments:
        - input (list of values): list of inputs (2D np.arrays), one for each Process in kwProcesses
            [??CORRECT: one item for each originMechanism (Mechanisms in the 1st set of self.executionGraph)]
            (default: variableInstanceDefault for the first Mechanism in each Process)
        - params (dict):
            + kwProcesses (list): (default: a single instance of the default Process)
            + kwController (list): (default: DefaultController)
            + kwEnableControl (bool): (default: False)
                specifies whether the controller is called during system execution
            + MONITORED_OUTPUT_STATES (list): (default: PRIMARY_OUTPUT_STATES)
                specifies the outputStates of the terminal mechanisms in the System
                    to be monitored by ControlMechanism
                this specification is overridden by any in ControlMechanism.params[] or Mechanism.params[]
                    or if None is specified for MONITORED_OUTPUT_STATES in the outputState itself
                each item must be one of the following:
                    + Mechanism or OutputState (object)
                    + Mechanism or OutputState name (str)
                    + (Mechanism or OutputState specification, exponent, weight) (tuple):
                        + mechanism or outputState specification (Mechanism, OutputState, or str):
                            referenceto Mechanism or OutputState object or the name of one
                            if a Mechanism ref, exponent and weight will apply to all outputStates of that mechanism
                        + exponent (int):  will be used to exponentiate outState.value when computing EVC
                        + weight (int): will be used to multiplicative weight outState.value when computing EVC
                    + MonitoredOutputStatesOption (AutoNumber enum):
                        + PRIMARY_OUTPUT_STATES:  monitor only the primary (first) outputState of the Mechanism
                        + ALL_OUTPUT_STATES:  monitor all of the outputStates of the Mechanism
                        Notes:
                        * this option applies to any mechanisms in the list for which no outputStates are listed;
                        * it is overridden for any mechanism for which outputStates are explicitly listed
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

    Instantiation:
        A System can be instantiated in one of two ways:
            [TBI: - Calling the run() function, which instantiates a default System]
            - by calling System(<args>)
        A System is instantiated by assigning:
            - the Mechanisms in all of the Processes in kwProcesses to a graph and an executionGraph
            - the executionGraph (with cyclic dependencies removed) is topologically sorted into
                 a sequentially ordered list of sets containing mechanisms to be executed at the same time
            - each input in it's input list to the first Mechanism of each Process
            - the outputs of terminal Mechanisms in the executionGraph System.outputState(s)
                (terminal mechanisms are ones that do not project to any other mechanisms in the System)

    SystemRegistry:
        All Processes are registered in ProcessRegistry, which maintains a dict for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Processes can be named explicitly (using the name='<name>' argument).  If this argument is omitted,
        it will be assigned "Mapping" with a hyphenated, indexed suffix ('Mapping-n')

# DOCUMENTATION: UPDATE Execution BELOW
    Execution:
        - System.execute calls mechanism.update_states_and_execute for each mechanism in its configuration in sequence
            - input specified as arg in execution of Process is provided as input to first mechanism in configuration
            - output of last mechanism in configuration is assigned as Process.ouputState.value
            - DefaultController is executed before execution of each mechanism in the configuration
            - notes:
                * the same mechanism can be listed more than once in a configuration, inducing recurrent processing
                * if it is the first mechanism, it will receive its input from the Process only once (first execution)
                [TBI: add option to allow input to be provided every time mechanism it executed]
            - Process.ouputState.value receives Mechanism.outputState.value of last mechanism in the configuration

    Class attributes:
        + functionCategory (str): kwProcessFunctionCategory
        + className (str): kwProcessFunctionCategory
        + suffix (str): " <kwMechanismFunctionCategory>"
        + registry (dict): ProcessRegistry
        + classPreference (PreferenceSet): ProcessPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault = inputValueSystemDefault                     # Used as default input value to Process)
        + paramClassDefaults = {kwProcesses: [Mechanism_Base.defaultMechanism],
                                kwController: DefaultController,
                                kwTimeScale: TimeScale.TRIAL}


    Class methods:
        - validate_variable(variable, context):  insures that variable is 3D np.array (one 2D for each Process)
        - instantiate_attributes_before_function(context):  calls self.instantiate_graph
        - instantiate_function(context): validates only if self.prefs.paramValidationPref is set
        - instantiate_graph(inputs, context):  instantiates Processes in self.process and constructs execution_list
        - identify_origin_and_terminal_mechanisms():  assign self.originMechanisms and self.terminalMechanisms
        - assign_output_states():  assign outputStates of System (currently = terminalMechanisms)
        - execute(inputs, time_scale, context):  executes Mechanisms in order specified by execution_list
        - variableInstanceDefaults(value):  setter for variableInstanceDefaults;  does some kind of error checking??

    Instance attributes:
        + processes (list):  an ordered list of processes
            derived from params[kwProcesses], possibly appended by EVCMechanism (with prediction processes)
            used with self.inputs to constsruct self.process_tuples
        + processList (SystemProcessList): provides access to (process, input) tuples
            derived from self.inputs and self.processes (params[kwProcesses])
            used to construct self.executionGraph and execute the System
            (default: a single instance of the default Process)
        + phaseSpecMax (int) - maximum phaseSpec for all Mechanisms in System
    [TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each process in self.processes
        + output (list): containts Process.ouput for each process in self.processes
    ]
        [TBI: + inputs (list): each item is the Process.input object for the corresponding Process in self.processes]
        [TBI: + outputs (list): each item is the Process.output object for the corresponding Process in self.processes]
        + graph (OrderedDict): each entry specifies a set of depenencies <Receiver>: {sender, sender...} dependencies;
            - the key of each entry is a mech_tuple
            - the value is a set of mech_tuples that send projections to the receiver (key);
            - if a key (receiver) has no dependents, its value is an empty set
        + executionGraph (OrderedDict): a hiearchically organized subset (toposort) of the graph that is acyclic;
            - it does not include any recurrent dependencies (ones that would introduce a cycle);
            - it is built by:
                sequentially beginning at the origin of each process,
                traversing all projections from each mechanism in the process (in order of the process' configuration)
                entering each mechanism "encountered" in the dependency set of the previous (sender) mechanism
                using toposort to test whether the dependency has introduced a cycle
                eliminating the dependent from the executionGraph if it has introduced a cycle
        + execution_sets (list of sets):
            each set contains mechanism to be executed at the same time;
            the sets are ordered in the sequence with which they should be executed
        + execute_list (list of Mechanisms):  a list of Mechanisms in the order they should be executed;
            Note: the list is a random sample subject to the constraints of ordering in self.execute_sets
        [TBI: + originMechanisms (list):  Mechanism objects without projections from any other Mechanisms in the System]
        + mechanisms (list): points to allMechanisms.mechanisms (see below)
        + mechanismsDict (dict): dict of Mechanism:Process entries for all Mechanisms in the System
            the key for each entry is a Mechanism object
            the value of each entry is a list of processes (since mechanisms can be in several Processes)
        Note: the following attributes use lists of tuples (mechanism, runtime_param, phaseSpec) and SystemMechanismList
              - the <type>_mech_tuples lists are comprised of tuples defined in the Process configurations;
                  these are used because runtime_params and phaseSpec are attributes that need
                  to be able to be specified differently for the same mechanism in different contexts
                  and thus are not easily managed as mechanism attributes
              - <type>MechanismLists point to SystemMechanismsList objects, that provide access to information
                  about the mechanism <type> listed in <type>_mech_tuples (i.e., the mechanisms, names, etc.)
        + all_mech_tuples (list):  list of all mech_tuples in the system (and that serve as keys in self.graph
        + allMechanisms (SystemMechanismsList):  list of (mechanism object, runtime_params dict, phaseSpec int) tuples
        + origin_mech_tuples (list):  mechanisms that don't receive projections from any other mechanisms in the System
        + originMechanisms (SystemMechanismsList): access to information about mechanisms in the origin_mech_tuples list
        + terminal_mech_tuples (list):  mechanisms that don't project to any other mechanisms in the System
        + terminalMechanisms (SystemMechanismsList): access to information about mechanisms in terminal_mech_tuples
            Note: the outputStates of the System's terminal mechanisms comprise the output values for System.output
        + monitoring_mech_tuples (list):  mechanism tuples for MonitoringMechanisms in the system (used for learning)
        + monitoringMechanisms (SystemMechanismsList): access to information about mechanisms in monitoring_mech_tuples
        + learning_mech_tuples (list):  mechanism tuples for LearningMechanisms in the system (used for learning)
        + monitoringMechanisms (SystemMechanismsList): access to information about mechanisms in learning_mech_tuples
        + control_mech_tuples (list):  mechanism tuples ControlMechanisms in the system
        + controlMechanisms (SystemMechanismsList): access to information about mechanisms in control_mech_tuples
        [TBI: + controller (ControlMechanism): the control mechanism assigned to the System]
            (default: DefaultController)
        + value (3D np.array):  each 2D array item the value (output) of the corresponding Process in kwProcesses
        + phaseSpecMax (int):  phase of last (set of) ProcessingMechanism(s) to be executed in the system
        + numPhases (int):  number of phases for system (= phaseSpecMax + 1)
        + initial_values (dict):  dict of initial values for all mechanisms designated as INITIALIZE_CYCLE
            in their mechanism.systems attribute;  for each entry:
            - the key is a mechanism object
            - the value is a number, list or np.array that must be compatible with mechanism.value
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
    # These will override those specified in CategoryDefaultPreferences
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemCustomClassPreferences',
    #     kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Use inputValueSystemDefault as default input to process
    # variableClassDefault = inputValueSystemDefault
    variableClassDefault = None

    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({kwTimeScale: TimeScale.TRIAL})

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 processes=None,
                 initial_values=None,
                 controller=SystemDefaultControlMechanism,
                 enable_controller=False,
                 monitored_output_states=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign category-level preferences, register category, call super.__init__ (that instantiates configuration)

        Args:
            default_input_value:
            processes:
            initial_values:
            controller:
            enable_controller:
            monitored_output_states:
            params:
            name:
            prefs:
            context:
        """
        # MODIFIED 9/20/16 NEW:  replaced above with None
        processes = processes or []
        monitored_output_states = monitored_output_states or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]
        # MODIFIED 9/20/16 END

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(processes=processes,
                                                 initial_values=initial_values,
                                                 controller=controller,
                                                 enable_controller=enable_controller,
                                                 monitored_output_states=monitored_output_states,
                                                 params=params)

        self.configuration = NotImplemented
        self.outputStates = {}
        self.phaseSpecMax = 0
        self.function = self.execute

        register_category(entry=self,
                          base_class=System_Base,
                          name=name,
                          registry=SystemRegistry,
                          context=context)

        if not context:
            # context = kwInit + self.name
            context = kwInit + self.name + kwSeparator + kwSystemInit

        super().__init__(variable_default=default_input_value,
                         param_defaults=params,
                         name=self.name,
                         prefs=prefs,
                         context=context)

        # Get/assign controller

        # Controller is DefaultControlMechanism
        from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.DefaultControlMechanism import DefaultControlMechanism
        if self.paramsCurrent[kwController] is DefaultControlMechanism:
            # Get DefaultController from MechanismRegistry
            from PsyNeuLink.Functions.Mechanisms.Mechanism import MechanismRegistry
            self.controller = list(MechanismRegistry[kwDefaultControlMechanism].instanceDict.values())[0]
        # Controller is not DefaultControlMechanism
        else:
            # Instantiate specified controller
            self.controller = self.paramsCurrent[kwController](params={SYSTEM: self})

        # Check whether controller has inputs, and if not then disable
        try:
            has_input_states = bool(self.controller.inputStates)
        except:
            has_input_states = False
        if not has_input_states:
            # If controller was enabled (and verbose is set), warn that it has been disabled
            if self.enable_controller and self.prefs.verbosePref:
                print("{} for {} has no inputStates, so controller will be disabled".
                      format(self.controller.name, self.name))
            self.enable_controller = False


        # Compare phaseSpecMax with controller's phaseSpec, and assign default if it is not specified
        try:
            # Get phaseSpec from controller
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
        #           format(self.name, self.mechanismNames.__str__().strip("[]")))

    def validate_variable(self, variable, context=None):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        :param variable:
        :param context:
        :return:
        """

        super(System_Base, self).validate_variable(variable, context)

        # # MODIFIED 6/26/16 OLD:
        # # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        # self.variable = convert_to_np_array(self.variable, 2)
# FIX:  THIS CURRENTLY FAILS:
        # # MODIFIED 6/26/16 NEW:
        # # Force System variable specification to be a 3D array (to accommodate input states for each Process):
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 3)
        # self.variable = convert_to_np_array(self.variable, 3)
        # MODIFIED 10/2/16 NEWER:
        # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        if variable is None:
            return
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Validate controller, processes and initial_values
        """
        super().validate_params(request_set=request_set, target_set=target_set, context=context)

        controller = target_set[kwController]
        if (not isinstance(controller, ControlMechanism_Base) and
                not (inspect.isclass(controller) and issubclass(controller, ControlMechanism_Base))):
            raise SystemError("{} (controller arg for \'{}\') is not a ControllerMechanism or subclass of one".
                              format(controller, self.name))

        for process in target_set[kwProcesses]:
            if not isinstance(process, Process):
                raise SystemError("{} (in processes arg for \'{}\') is not a Process object".format(process, self.name))

        for mech, value in target_set[kwInitialValues].items():
            if not isinstance(mech, Mechanism):
                raise SystemError("{} (key for entry in initial_values arg for \'{}\') "
                                  "is not a Mechanism object".format(mech, self.name))

    def instantiate_attributes_before_function(self, context=None):
        """Instantiate processes and graph

        These must be done before instantiate_function as the latter may be called during init for validation
        """
        self.instantiate_processes(inputs=self.variable, context=context)
        self.instantiate_graph(context=context)

    def instantiate_function(self, context=None):
        """Suppress validation of function

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a System object)
        - suppress validation (and attendant execution) of System execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in kwProcesses have already been validated

        :param context:
        :return:
        """

        if self.paramsCurrent[FUNCTION] != self.execute:
            print("System object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[FUNCTION], FUNCTION)
            self.paramsCurrent[FUNCTION] = self.execute

        # If validation pref is set, instantiate and execute the System
        if self.prefs.paramValidationPref:
            super(System_Base, self).instantiate_function(context=context)
        # Otherwise, just set System output info to the corresponding info for the last mechanism(s) in self.processes
        else:
            self.value = self.processes[-1].outputState.value

# FIX:
#     ** PROBLEM: self.value IS ASSIGNED TO variableInstanceDefault WHICH IS 2D ARRAY,
        # BUT PROJECTION EXECUTION FUNCTION TAKES 1D ARRAY
#         Assign projection from Process (self.value) to inputState of the first mechanism in the configuration
#     **?? WHY DO THIS, IF SELF.VALUE HAS BEEN ASSIGNED AN INPUT VALUE, AND PROJECTION IS PROVIDING INPUT TO MECHANISM??
#         Assigns variableInstanceDefault to variableInstanceDefault of first mechanism in configuration

# FIX: ALLOW Projections (??ProjectionTiming TUPLES) TO BE INTERPOSED BETWEEN MECHANISMS IN CONFIGURATION
# FIX: AUGMENT LinearMatrix TO USE FULL_CONNECTIVITY_MATRIX IF len(sender) != len(receiver)

    def instantiate_processes(self, inputs=None, context=None):
        """Instantiate processes of system

        Use self.processes (populated by self.paramsCurrent[kwProcesses] in Function.assign_args_to_param_dicts
        If self.processes is empty, instantiate default Process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        If inputs is specified, check that it's length equals the number of processes
        If inputs is not specified, compose from the input for each Process (value specified or, if None, default)
        Note: specification of input for system takes precedence over specification for processes

        # ??STILL THE CASE, OR MOVED TO instantiate_graph:
        Iterate through Process.mechanismList for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.executionGraph
        """

        self.variable = []
        self.mechanismsDict = {}
        self.all_mech_tuples = []
        self.allMechanisms = SystemMechanismsList(self, self.all_mech_tuples)

        # Get list of processes specified in arg to init, possiblly appended by EVCMechanism (with prediction processes)
        processes_spec = self.processes

        # Assign default Process if kwProcess is empty, or invalid
        if not processes_spec:
            from PsyNeuLink.Functions.Process import Process_Base
            processes_spec.append((Process_Base(), None))

        # If inputs to system are specified, number must equal number of processes with origin mechanisms
        if not inputs is None and len(inputs) != len(self.originMechanisms):
            raise SystemError("Number of inputs ({0}) must equal number of processes ({1}) in {} ".
                              format(len(inputs), len(self.originMechanisms)),
                              self.name)

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS

        for i in range(len(processes_spec)):

            # Convert all entries to (process, input) tuples, with None as filler for absent inputs
            if not isinstance(processes_spec[i], tuple):
                processes_spec[i] = (processes_spec[i], None)

            if inputs is None:
                # FIX: ASSIGN PROCESS INPUTS TO SYSTEM INPUTS
                process = processes_spec[i][PROCESS]
                process_input = []
                for process_input_state in process.processInputStates:
                    process_input.extend(process_input_state.value)
                processes_spec[i] = (process, process_input)
            # If input was provided on command line, assign that to input item of tuple
            else:
                # Assign None as input to processes implemented by controller (controller provides their input)
                #    (e.g., prediction processes implemented by EVCMechanism)
                if processes_spec[i][PROCESS].isControllerProcess:
                    processes_spec[i] = (processes_spec[i][PROCESS], None)
                else:
                    # Replace input item in tuple with one from variable
                    processes_spec[i] = (processes_spec[i][PROCESS], inputs[i])

            # Validate input
            if (not processes_spec[i][PROCESS_INPUT] is None and
                    not isinstance(processes_spec[i][PROCESS_INPUT],(numbers.Number, list, np.ndarray))):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, processes_spec[i][PROCESS_INPUT]))

            process = processes_spec[i][PROCESS]
            input = processes_spec[i][PROCESS_INPUT]
            self.variable.append(input)

            # If process item is a Process object, assign input as default
            if isinstance(process, Process):
                if not input is None:
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

            # # process should now be a Process object;  assign to processList
            # self.processList.append(process)

            # Assign the Process a reference to this System
            process.system = self

            # Get max of Process phaseSpecs
            self.phaseSpecMax = int(max(math.floor(process.phaseSpecMax), self.phaseSpecMax))

            # FIX: SHOULD BE ABLE TO PASS INPUTS HERE, NO?  PASSED IN VIA VARIABLE, ONE FOR EACH PROCESS
            # FIX: MODIFY instantiate_configuration TO ACCEPT input AS ARG
            # NEEDED?? WASN"T IT INSTANTIATED ABOVE WHEN PROCESS WAS INSTANTIATED??
            # process.instantiate_configuration(self.variable[i], context=context)

            # Iterate through mechanism tuples in Process' mechanismList
            #     to construct self.all_mech_tuples and mechanismsDict
            # FIX: ??REPLACE WITH:  for sender_mech_tuple in process.mechanismList
            for sender_mech_tuple in process.mechanismList:

                sender_mech = sender_mech_tuple[MECHANISM]

                # THIS IS NOW DONE IN instantiate_graph
                # # Add system to the Mechanism's list of systems of which it is member
                # if not self in sender_mech_tuple[MECHANISM].systems:
                #     sender_mech.systems[self] = INTERNAL

                # Assign sender mechanism entry in self.mechanismsDict, with mech_tuple as key and its Process as value
                #     (this is used by Process.instantiate_configuration() to determine if Process is part of System)
                # If the sender is already in the System's mechanisms dict
                if sender_mech_tuple[MECHANISM] in self.mechanismsDict:
                    existing_mech_tuple = self.allMechanisms.get_tuple_for_mech(sender_mech)
                    if not sender_mech_tuple is existing_mech_tuple:
                        # Contents of tuple are the same, so use the tuple in allMechanisms
                        if (sender_mech_tuple[PHASE_SPEC] == existing_mech_tuple[PHASE_SPEC] and
                                    sender_mech_tuple[PROCESS_INPUT] == existing_mech_tuple[PROCESS_INPUT]):
                            pass
                        # Contents of tuple are different, so raise exception
                        else:
                            if sender_mech_tuple[PHASE_SPEC] != existing_mech_tuple[PHASE_SPEC]:
                                offending_tuple_field = 'phase'
                                offending_value = PHASE_SPEC
                            else:
                                offending_tuple_field = 'input'
                                offending_value = PROCESS_INPUT
                            raise SystemError("The same mechanism in different processes must have the same parameters:"
                                              "the {} ({}) for {} in {} does not match the value({}) in {}".
                                              format(offending_tuple_field,
                                                     sender_mech_tuple[MECHANISM],
                                                     sender_mech_tuple[offending_value],
                                                     process,
                                                     existing_mech_tuple[offending_value],
                                                     self.mechanismsDict[sender_mech_tuple[MECHANISM]]
                                                     ))
                    # Add to entry's list
                    self.mechanismsDict[sender_mech].append(process)
                else:
                    # Add new entry
                    self.mechanismsDict[sender_mech] = [process]
                if not sender_mech_tuple in self.all_mech_tuples:
                    self.all_mech_tuples.append(sender_mech_tuple)

            process.mechanisms = ProcessMechanismsList(process)

        self.variable = convert_to_np_array(self.variable, 2)

        # Instantiate processList using process_tuples, and point self.processes to it
        # Note: this also points self.params[kwProcesses] to self.processes
        self.process_tuples = processes_spec
        self.processList = SystemProcessList(self)
        self.processes = self.processList.processes

    def instantiate_graph(self, context=None):
        # DOCUMENTATION: EXPAND BELOW
        """Construct graphs (full and acyclic) of system

        graph -> full graph
        execution_graph -> acyclic graph
        Prune projections from processes or mechanisms in processes not in the system
        Ignore feedback projections in construction of dependency_set
        Assign self to each mechanism.systems with mechanism's status as value:
            - ORIGIN,
            - TERMINAL
            - SINGLETON
            - INTERNAL
            - CYCLE (receives recurrent projection), and INITIALIZE_CYCLE (initialize)
            - MONITORING
            - CONTROL

        Construct self.mechanismsList, self.mech_tuples, self.allMechanisms
        Validate initial_values

        """

        # Use to recursively traverse processes
        def build_dependency_sets_by_traversing_projections(sender_mech):

            # Delete any projections to mechanism from processes or mechanisms in processes not in current system
            for input_state in sender_mech.inputStates.values():
                for projection in input_state.receivesFromProjections:
                    sender = projection.sender.owner
                    system_processes = self.processes
                    if isinstance(sender, Process):
                        if not sender in system_processes:
                            del projection
                    elif not all(sender_process in system_processes for sender_process in sender.processes):
                        del projection

            # If sender_mech has no projections left, raise exception
            if not any(any(projection for projection in input_state.receivesFromProjections)
                       for input_state in sender_mech.inputStates.values()):
                raise SystemError("{} only receives projections from other processes or mechanisms not"
                                  " in the current system ({})".format(sender_mech.name, self.name))

            # Assign as TERMINAL (or SINGLETON) if it has no outgoing projections and is not a Comparator or
            #     it projects only to Comparator(s)
            # Note:  SINGLETON is assigned if mechanism is already a TERMINAL;  indicates that it is both
            #        an ORIGIN AND A TERMINAL and thus must be the only mechanism in its process
            if (not isinstance(sender_mech, (MonitoringMechanism_Base, ControlMechanism_Base)) and
                    all(all(isinstance(projection.receiver.owner, (MonitoringMechanism_Base, ControlMechanism_Base))
                            for projection in output_state.sendsToProjections)
                        for output_state in sender_mech.outputStates.values())):
                try:
                    if sender_mech.systems[self] is ORIGIN:
                        sender_mech.systems[self] = SINGLETON
                    else:
                        sender_mech.systems[self] = TERMINAL
                except KeyError:
                    sender_mech.systems[self] = TERMINAL
                return

            for outputState in sender_mech.outputStates.values():

                for projection in outputState.sendsToProjections:
                    receiver = projection.receiver.owner
                    receiver_tuple = self.allMechanisms.get_tuple_for_mech(receiver)

                    try:
                        self.graph[receiver_tuple].add(self.allMechanisms.get_tuple_for_mech(sender_mech))
                    except KeyError:
                        self.graph[receiver_tuple] = {self.allMechanisms.get_tuple_for_mech(sender_mech)}

                    # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                    # Do not include dependency (or receiver on sender) in execution_graph for this projection
                    #  and end this branch of the traversal if the receiver has already been encountered,
                    #  but do mark for initialization
                    # Notes:
                    # * This is because it is a feedback connection, which introduces a cycle into the graph
                    #     that precludes use of toposort to determine order of execution;
                    #     however, the feedback projection will still be used during execution
                    #     so the sending mechanism should be designated as INITIALIZE_CYCLE
                    # * Check for receiver mechanism and not its tuple,
                    #     since the same mechanism can appear in more than one tuple (e.g., with different phases)
                    #     and would introduce a cycle irrespective of the tuple in which it appears in the graph
                    # FIX: MODIFY THIS TO (GO BACK TO) USING if receiver_tuple in self.executionGraph
                    # FIX  BUT CHECK THAT THEY ARE IN DIFFERENT PHASES
                    if receiver in self.execution_graph_mechs:
                        # Try assigning receiver as dependent of current mechanism and test toposort
                        try:
                            # If receiver_tuple already has dependencies in its set, add sender_mech to set
                            if self.executionGraph[receiver_tuple]:
                                self.executionGraph[receiver_tuple].add(self.allMechanisms.get_tuple_for_mech(sender_mech))
                            # If receiver_tuple set is empty, assign sender_mech to set
                            else:
                                self.executionGraph[receiver_tuple] = {self.allMechanisms.get_tuple_for_mech(sender_mech)}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.executionGraph))
                        # If making receiver dependent on sender produced a cycle (feedback loop), remove from graph
                        except ValueError:
                            self.executionGraph[receiver_tuple].remove(self.allMechanisms.get_tuple_for_mech(sender_mech))
                            # Assign sender_mech INITIALIZE_CYCLE as system status if not ORIGIN or not yet assigned
                            if not sender_mech.systems or not (sender_mech.systems[self] in {ORIGIN, SINGLETON}):
                                sender_mech.systems[self] = INITIALIZE_CYCLE
                            if not (receiver.systems[self] in {ORIGIN, SINGLETON}):
                                receiver.systems[self] = CYCLE
                            continue

                    else:
                        # Assign receiver as dependent on sender mechanism
                        try:
                            # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                            # FIX: DOES THAT SCREW UP ORIGINS?
                            self.executionGraph[receiver_tuple].add(self.allMechanisms.get_tuple_for_mech(sender_mech))
                        except KeyError:
                            self.executionGraph[receiver_tuple] = {self.allMechanisms.get_tuple_for_mech(sender_mech)}

                    if not sender_mech.systems:
                        sender_mech.systems[self] = INTERNAL

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver)

        from collections import OrderedDict
        self.graph = OrderedDict()
        self.executionGraph = OrderedDict()

        for process in self.processes:
            first_mech = process.firstMechanism
            # Treat as ORIGIN if ALL projections to the first mechanism in the process are from:
            #    - the process itself (ProcessInputState
            #    - another mechanism in the in process (i.e., feedback projections from *within* the process)
            #    - mechanisms from other process for which it is an origin
            # Notes:
            # * This precludes a mechanism that is an ORIGIN of a process from being an ORIGIN for the system
            #       if it receives any projections from any other mechanisms in the system (including other processes)
            #       other than ones in processes for which it is also their ORIGIN
            # * This does allow a mechanism to be the ORIGIN (but *only* the ORIGIN) for > 1 process in the system
            if all(
                    all(
                            # All projections must be from a process (i.e., ProcessInputState) to which it belongs
                            projection.sender.owner in self.processes or
                            # or from mechanisms within its own process (e.g., [a, b, a])
                            projection.sender.owner in list(process.mechanisms) or
                            # or from mechanisms in oher processes for which it is also an ORIGIN ([a, b, a], [a, c, a])
                            all(ORIGIN in first_mech.processes[proc] for proc in projection.sender.owner.processes)
                        for projection in input_state.receivesFromProjections)
                    for input_state in first_mech.inputStates.values()):
                # Assign its set value as empty, marking it as a "leaf" in the graph
                mech_tuple = self.allMechanisms.get_tuple_for_mech(first_mech)
                self.graph[mech_tuple] = set()
                self.executionGraph[mech_tuple] = set()
                first_mech.systems[self] = ORIGIN

            build_dependency_sets_by_traversing_projections(first_mech)

        # Print graph
        if self.verbosePref:
            print("In the system graph for \'{}\':".format(self.name))
            for receiver_mech_tuple, dep_set in self.executionGraph.items():
                mech = receiver_mech_tuple[MECHANISM]
                if not dep_set:
                    print("\t\'{}\' is an {} mechanism".
                          format(mech.name, mech.systems[self]))
                else:
                    status = mech.systems[self]
                    if status is TERMINAL:
                        status = 'a ' + status
                    elif status in {INTERNAL, INITIALIZE_CYCLE}:
                        status = 'an ' + status
                    print("\t\'{}\' is {} mechanism that receives projections from:".format(mech.name, status))
                    for sender_mech_tuple in dep_set:
                        print("\t\t\'{}\'".format(sender_mech_tuple[MECHANISM].name))

        # For each mechanism (represented by its tuple) in the graph, add entry to relevant list(s)
        # Note: ignore mechanisms belonging to controllerProcesses (e.g., instantiated by EVCMechanism)
        #       as they are for internal use only
        self.origin_mech_tuples = []
        self.terminal_mech_tuples = []
        self.recurrent_init_mech_tuples = []
        self.control_mech_tuples = []
        self.monitoring_mech_tuples = []

        for mech_tuple in self.executionGraph:
            mech = mech_tuple[MECHANISM]
            if mech.systems[self] in {ORIGIN, SINGLETON}:
                for process, status in mech.processes.items():
                    if process.isControllerProcess:
                        continue
                    self.origin_mech_tuples.append(mech_tuple)
                    break
            if mech_tuple[MECHANISM].systems[self] in {TERMINAL, SINGLETON}:
                for process, status in mech.processes.items():
                    if process.isControllerProcess:
                        continue
                    self.terminal_mech_tuples.append(mech_tuple)
                    break
            if mech_tuple[MECHANISM].systems[self] in {INITIALIZE_CYCLE}:
                for process, status in mech.processes.items():
                    if process.isControllerProcess:
                        continue
                    self.recurrent_init_mech_tuples.append(mech_tuple)
                    break
            if isinstance(mech_tuple[MECHANISM], ControlMechanism_Base):
                if not mech_tuple[MECHANISM] in self.control_mech_tuples:
                    self.control_mech_tuples.append(mech_tuple)
            if isinstance(mech_tuple[MECHANISM], MonitoringMechanism_Base):
                if not mech_tuple[MECHANISM] in self.monitoring_mech_tuples:
                    self.monitoring_mech_tuples.append(mech_tuple)

        self.originMechanisms = SystemMechanismsList(self, self.origin_mech_tuples)
        self.terminalMechanisms = SystemMechanismsList(self, self.terminal_mech_tuples)
        self.recurrentInitMechanisms = SystemMechanismsList(self, self.recurrent_init_mech_tuples)
        self.controlMechanisms = SystemMechanismsList(self, self.control_mech_tuples)
        self.monitoringMechanisms = SystemMechanismsList(self, self.monitoring_mech_tuples)

        try:
            self.execution_sets = list(toposort(self.executionGraph))
        except ValueError as e:
            if 'Cyclic dependencies exist' in e.args[0]:
                # if self.verbosePref:
                # print('{} has feedback connections; be sure that the following items are properly initialized:'.
                #       format(self.name))
                raise SystemError("PROGRAM ERROR: cycle (feedback loop) in {} not detected by instantiate_graph ".
                                  format(self.name))

        # Create instance of sequential (execution) list:
        self.execution_list = toposort_flatten(self.executionGraph, sort=False)

        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITALIZE HAVE AN INITIAL_VALUES ENTRY
        for mech, value in self.initial_values.items():
            if not mech in self.execution_graph_mechs:
                raise SystemError("{} (entry in initial_values arg) is not a Mechanism in \'{}\'".
                                  format(mech.name, self.name))
            if not iscompatible(value, mech.variable):
                raise SystemError("{} (in initial_values arg for \'{}\') is not a valid value for {}".
                                  format(value, self.name, append_type_to_name(self)))

    def assign_output_states(self):
        """Assign outputStates for System (the values of which will comprise System.value)

        Note:
        * Current implementation simply assigns terminal mechanisms as outputStates
        * This method is included so that sublcasses and/or future versions can override it to make custom assignments

        """
        for mech in self.terminalMechanisms.mechanisms:
            self.outputStates[mech.name] = mech.outputStates

    def initialize(self):
        # Initialize feedback mechanisms
        # FIX:  INITIALIZE PROCESS INPUTS??
        # FIX: CHECK THAT ALL MECHANISMS ARE INITIALIZED FOR WHICH mech.system[SELF]==INITIALIZE
        # FIX: ADD OPTION THAT IMPLEMENTS/ENFORCES INITIALIZATION
        for mech, value in self.initial_values.items():
            mech.initialize(value)

    def execute(self,
                inputs=None,
                time_scale=None,
                context=None):
# DOCUMENT: NEEDED -- INCLUDE HANDLING OF phaseSpec
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

        if not context:
            context = kwExecuting + self.name
        report_system_output = self.prefs.reportOutputPref and context and kwExecuting in context
        if report_system_output:
            report_process_output = any(process.reportOutputPref for process in self.processes)

        self.timeScale = time_scale or TimeScale.TRIAL

        #region ASSIGN INPUTS TO PROCESSES
        # Assign each item of input to the value of a Process.input_state which, in turn, will be used as
        #    the input to the mapping projection to the first (origin) Mechanism in that Process' configuration
        if inputs is None:
            pass
        else:
            # # MODIFIED 10/8/16 OLD:
            # if len(inputs) != len(list(self.originMechanisms)):
            #     raise SystemError("Number of inputs ({0}) to {1} does not match its number of origin Mechanisms ({2})".
            #                       format(len(inputs), self.name,  len(list(self.originMechanisms)) ))
            # # MODIFIED 10/8/16 NEW:
            # if (isinstance(inputs, np.ndarray) and np.size(inputs) != len(list(self.originMechanisms)) or
            #     not isinstance(inputs, np.ndarray) and len(inputs) != len(list(self.originMechanisms))):
            #         raise SystemError("Number of inputs ({0}) to {1} does not match its number of origin Mechanisms ({2})".
            #                           format(len(inputs), self.name,  len(list(self.originMechanisms)) ))
            # MODIFIED 10/8/16 NEWER:
            num_inputs = np.size(inputs,0)
            num_origin_mechs = len(list(self.originMechanisms))
            if num_inputs != num_origin_mechs:
                # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
                num_inputs = np.size(inputs)
                if isinstance(inputs, np.ndarray) and inputs.dtype is np.dtype('O') and num_inputs == num_origin_mechs:
                    pass
                else:
                    raise SystemError("Number of inputs ({0}) to {1} does not match "
                                      "its number of origin Mechanisms ({2})".
                                      format(num_inputs, self.name,  num_origin_mechs ))
            # MODIFIED 10/8/16 END
            for i in range(num_inputs):
                input = inputs[i]
                process = self.processes[i]

                # Make sure there is an input, and if so convert it to 2D np.ndarray (required by Process
                if input is None or input is NotImplemented:
                    continue
                else:
                    # Assign input as value of corresponding Process inputState
                    process.assign_input_values(input=input, context=context)
        self.inputs = inputs
        #endregion

        if report_system_output:
            self.report_system_initiation()


        #region EXECUTE EACH MECHANISM

        # Execute each Mechanism in self.execution_list, in the order listed
        for i in range(len(self.execution_list)):

            mechanism, params, phase_spec = self.execution_list[i]

            if report_system_output and report_process_output:
                for process, status in mechanism.processes.items():
                    if status in {ORIGIN, SINGLETON} and process.reportOutputPref:
                        process.report_process_initiation()

            # Only update Mechanism on time_step(s) determined by its phaseSpec (specified in Mechanism's Process entry)
# FIX: NEED TO IMPLEMENT FRACTIONAL UPDATES (IN Mechanism.update()) FOR phaseSpec VALUES THAT HAVE A DECIMAL COMPONENT
            if phase_spec == (CentralClock.time_step % self.numPhases):
                # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
                mechanism.execute(time_scale=self.timeScale,
                # mechanism.execute(time_scale=time_scale,
                                 runtime_params=params,
                                 context=context)

                # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR MECHANISM ITSELF
                if report_system_output:
                    if report_process_output:
                        for process, status in mechanism.processes.items():
                            if status is TERMINAL and process.reportOutputPref:
                                process.report_process_completion()

            if not i:
                # Zero input to first mechanism after first run (in case it is repeated in the configuration)
                # IMPLEMENTATION NOTE:  in future version, add option to allow Process to continue to provide input
                # # MODIFIED 10/2/16 OLD:
                # self.variable = self.variable * 0
                # # MODIFIED 10/2/16 NEW:
                # self.variable = self.inputs * 0
                # MODIFIED 10/2/16 NEWER:
                self.variable = convert_to_np_array(self.inputs, 2) * 0
                # MODIFIED 10/2/16 END
            i += 1
        #endregion


        # region EXECUTE LEARNING FOR EACH PROCESS

        for process in self.processes:
            if process.learning and process.learning_enabled:
                process.execute_learning(context=context)
        # endregion


        #region EXECUTE CONTROLLER

# FIX: 1) RETRY APPENDING TO EXECUTE LIST AND COMPARING TO THIS VERSION
# FIX: 2) REASSIGN INPUT TO SYSTEM FROM ONE DESIGNATED FOR EVC SIMULUS (E.G., StimulusPrediction)

        # Only call controller if this is not a controller simulation run (to avoid infinite recursion)
        if not kwEVCSimulation in context and self.enable_controller:
            try:
                if self.controller.phaseSpec == (CentralClock.time_step % self.numPhases):
                    self.controller.execute(time_scale=TimeScale.TRIAL,
                                            runtime_params=None,
                                            context=context)
                    if report_system_output:
                        print("{0}: {1} executed".format(self.name, self.controller.name))

            except AttributeError:
                if not 'INIT' in context:
                    raise SystemError("PROGRAM ERROR: no controller instantiated for {0}".format(self.name))
        #endregion

        # Report completion of system execution and value of designated outputs
        if report_system_output:
            self.report_system_completion()

        return self.terminalMechanisms.outputStateValues

    def mechs_in_graph(self):
        return list(m[MECHANISM] for m in self.executionGraph)

    def mech_names_in_graph(self):
        return list(m[MECHANISM].name for m in self.executionGraph)

    def mech_status_in_graph(self):
        return list(m[MECHANISM].systems for m in self.executionGraph)

    def report_system_initiation(self):

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        if CentralClock.time_step == 0:
            print("\n\'{}\'{} executing with: **** (time_step {}) ".
                  format(self.name, system_string, CentralClock.time_step))
            processes = list(process.name for process in self.processes)
            print("- processes: {}".format(processes))


        else:
            print("\n\'{}\'{} executing ********** (time_step {}) ".
                  format(self.name, system_string, CentralClock.time_step))

    def report_system_completion(self):

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        # Print output value of primary (first) outputState of each terminal Mechanism in System
        # IMPLEMENTATION NOTE:  add options for what to print (primary, all or monitored outputStates)
        print("\n\'{}\'{} completed ***********(time_step {})".format(self.name, system_string, CentralClock.time_step))
        for mech in self.terminal_mech_tuples:
            if mech[MECHANISM].phaseSpec == (CentralClock.time_step % self.numPhases):
                print("- output for {0}: {1}".format(mech[MECHANISM].name,
                                                     re.sub('[\[,\],\n]','',str(mech[MECHANISM].outputState.value))))

    class InspectOptions(AutoNumber):
        ALL = ()
        EXECUTION_SETS = ()
        EXECUTION_LIST = ()
        ATTRIBUTES = ()
        ALL_OUTPUTS = ()
        ALL_OUTPUT_LABELS = ()
        PRIMARY_OUTPUTS = ()
        PRIMARY_OUTPUT_LABELS = ()
        MONITORED_OUTPUTS = ()
        MONITORED_OUTPUT_LABELS = ()
        FLAT_OUTPUT = ()
        DICT_OUTPUT = ()


    def show(self, options=None):
        """Print execution_sets, execution_list, origin and terminal mechanisms, outputs, output labels
        """

        # # IMPLEMENTATION NOTE:  Stub for implementing options:
        # if options and self.InspectOptions.ALL_OUTPUT_LABELS in options:
        #     pass

        print ("\n---------------------------------------------------------")
        print ("\n{0}".format(self.name))

        # Print execution_sets (output of toposort)
        print ("\n\tExecution sets: ".format(self.name))
        for i in range(len(self.execution_sets)):
            print ("\t\tSet {0}:\n\t\t\t".format(i),end='')
            print("{ ",end='')
            for mech_tuple in self.execution_sets[i]:
                print("{0} ".format(mech_tuple[MECHANISM].name), end='')
            print("}")

        # Print execution_list sorted by phase and including EVC mechanism

        # Sort execution_list by phase
        sorted_execution_list = self.execution_list.copy()

        # Add controller to execution list for printing if enabled
        if self.enable_controller:
            sorted_execution_list.append((self.controller, None, self.controller.phaseSpec))

        # Sort by phaseSpec
        sorted_execution_list.sort(key=lambda mech_tuple: mech_tuple[PHASE_SPEC])

        print ("\n\tExecution list: ".format(self.name))
        phase = 0
        print("\t\tPhase {0}:".format(phase))
        for mech_tuple in sorted_execution_list:
            if mech_tuple[PHASE_SPEC] != phase:
                phase = mech_tuple[PHASE_SPEC]
                print("\t\tPhase {0}:".format(phase))
            print ("\t\t\t{0}".format(mech_tuple[MECHANISM].name))

        print ("\n\tOrigin mechanisms: ".format(self.name))
        for mech_tuple in self.originMechanisms.mech_tuples:
            print("\t\t{0} (phase: {1})".format(mech_tuple[MECHANISM].name, mech_tuple[PHASE_SPEC]))

        print ("\n\tTerminal mechanisms: ".format(self.name))
        for mech_tuple in self.terminalMechanisms.mech_tuples:
            print("\t\t{0} (phase: {1})".format(mech_tuple[MECHANISM].name, mech_tuple[PHASE_SPEC]))
            for output_state_name in mech_tuple[MECHANISM].outputStates:
                print("\t\t\t{0}".format(output_state_name))

        print ("\n---------------------------------------------------------")

    def inspect(self):
        """Return dictionary with attributes of system
               processes
               mechanisms
               originMechanisms
               terminalMechanisms
               intializeRecurrentProjections
               inputShape
               initializationShape
               outputValueShape
               numPhasesPerTrial
               monitoringMechanisms
               learningProjectionReceivers
               controlMechanisms
               controlProjectionsReceivers
        """

        input_array = []
        for mech in list(self.originMechanisms.mechanisms):
            input_array.append(mech.value)
        input_array = np.array(input_array)

        recurrent_init_array = []
        for mech in list(self.recurrentInitMechanisms.mechanisms):
            recurrent_init_array.append(mech.value)
        recurrent_init_array = np.array(recurrent_init_array)

        output_value_array = []
        for mech in list(self.terminalMechanisms.mechanisms):
            output_value_array.append(mech.outputValue)
        output_value_array = np.array(output_value_array)

        from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
        from PsyNeuLink.Functions.Projections.LearningSignal import LearningSignal
        learning_projections = []
        controlled_parameters = []
        for mech in list(self.mechanisms):
            for parameter_state in mech.parameterStates:
                try:
                    for projection in parameter_state.receivesFromProjections:
                        if isinstance(projection, ControlSignal):
                            controlled_parameters.append(parameter_state)
                except AttributeError:
                    pass
            for output_state in mech.outputStates:
                try:
                    for projection in output_state.sendsToProjections:
                        for parameter_state in projection.paramaterStates:
                            for sender in parameter_state.receivesFromProjections:
                                if isinstance(sender, LearningSignal):
                                    learning_projections.append(projection)
                except AttributeError:
                    pass

        inspect_dict = {
            'processes':self.processes,
            'mechanisms':self.mechanisms,
            'origin_mechanisms':self.originMechanisms.mechanisms,
            'terminal_mechanisms':self.terminalMechanisms.mechanisms,
            'recurrent_mechanisms':self.recurrentInitMechanisms,
            'control_mechanisms':self.controlMechanisms,
            'monitoring_mechanisms':self.monitoringMechanisms,
            'phases_per_trial':self.numPhases,
            'input_array':input_array,
            'recurrent_init_array':recurrent_init_array,
            'output_value_shape':output_value_array,
            'control_projections_receivers':controlled_parameters,
            'learning_projection_receivers':learning_projections
        }

        return inspect_dict

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

    @property
    def mechanisms(self):
        return self.allMechanisms.mechanisms

    @property
    def inputValue(self):
        return self.variable

    @property
    def numPhases(self):
        return self.phaseSpecMax + 1

    @property
    def execution_graph_mechs(self):
        """Mechanisms whose mech_tuples appear as keys in self.execution_graph

        Returns: list of mechanisms from mech_tuples in keys for execution_graph
        """
        return list(mech_tuple[0] for mech_tuple in self.executionGraph)
