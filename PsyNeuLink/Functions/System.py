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
from collections import UserList

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
        self.process_tuples = system.processes
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
        return list(item[MECHANISM].name for item in self.mechanisms)

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
    def __init__(self, system):
        self.mech_tuples = system.all_mech_tuples
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
           processes=[],
           controller=SystemDefaultControlMechanism,
           enable_controller=False,
           monitored_output_states=[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES],
           params=None,
           name=None,
           prefs:is_pref_set=None,
           context=None):

    """Return instance of System_Base

    If called with no arguments, returns System with default Process
    If called with a name string, uses it as the name for an instantiation of the System
    If a params dictionary is included, it is passed to the System (inclulding processes)

    See System_Base for description of args
    """

    # Called with descriptor keyword
    if not processes:
        processes = [process()]

    return System_Base(default_input_value=default_input_value,
                       processes=processes,
                       controller=controller,
                       enable_controller=enable_controller,
                       monitored_output_states=monitored_output_states,
                       params=params,
                       name=name,
                       prefs=prefs,
                       context=context)


class System_Base(System):
    # DOCUMENT: enable_controller option
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
        + processes (list of (process, input) tuples):  an ordered list of processes and corresponding inputs;
            derived from self.inputs and params[kwProcesses], and used to construct self.graph and execute the System
                 (default: a single instance of the default Process)
        + processList (list of processes): corresponds to processes in (process, input) tuples of self.processes
        + phaseSpecMax (int) - maximum phaseSpec for all Mechanisms in System
    [TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each process in self.processes
        + output (list): containts Process.ouput for each process in self.processes
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
        + mechanismsDict (dict): dict of Mechanism:Process entries for all Mechanisms in the System
            the key for each entry is a Mechanism object
            the value of each entry is a list of processes (since mechanisms can be in several Processes)
        Note: the following lists use (mechanism, runtime_param, phaseSpec) tuples
              that are defined in the Process configurations;  these are used because runtime_params and phaseSpec are
              attributes that need to be able to be specified differently for the same mechanism in different contexts
              and thus are not as easily managed as mechanism attributes
        + all_mech_tuples (list):  list of all mech_tuples in the System
            Notes:
            * each item is a (mechanism, runtime_params, phaseSpec) tuple
            * each tuple is an entry of Process.mechanismList for the Process in which the Mechanism occurs
            * each tuple serves as the key for the mechanism in self.graph
        + allMechanisms (SystemMechanisms):  list of (mechanism object, runtime_params dict, phaseSpec int) tuples
            for all mechanisms in the system
            Notes:
            * each item is a (mechanism, runtime_params, phaseSpec) tuple
            * each tuple is an entry of Process.mechanismList for the Process in which the Mechanism occurs
            * each tuple serves as the key for the mechanism in self.graph
            * <system>.mechanisms points to <system>.allMechanisms.mechanisms
        + origin_mech_tuples (list):  Mechanisms that don't receive projections from any other Mechanisms in the System
            Notes:
            * each item is a (mechanism, runtime_params, phaseSpec) tuple
            * each tuple is an entry of Process.mechanismList for the Process in which the Mechanism occurs
            * each tuple serves as the key for the mechanism in self.graph
        + originMechanisms (OriginMechanisms):  Mechanisms that don't receive projections from any other Mechanisms
            Notes:
            * this points to an OriginMechanism object that provides access to information about the originMechanisms
                 in the tuples of self.origin_mech_tuples
        + terminal_mech_tuples (list):  Mechanisms that don't project to any other Mechanisms in the System
            Notes:
            * each item is a (mechanism, runtime_params, phaseSpec) tuple
            * each tuple is an entry of Process.mechanismList for the Process in which the Mechanism occurs
            * each tuple serves as the key for the mechanism in self.graph
        + terminalMechanisms (TerminalMechanisms):  Mechanisms don't project to any other Mechanisms in the System
            Notes:
            * this points to a TerminalMechanism object that provides access to information about the terminalMechanisms
                in the tuples of self.terminal_mech_tuples
            * the outputStates of the System's terminal mechanisms comprise the output values for System.output
        [TBI: + controller (ControlMechanism): the control mechanism assigned to the System]
            (default: DefaultController)
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
                 controller=SystemDefaultControlMechanism,
                 enable_controller=False,
                 monitored_output_states=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign category-level preferences, register category, call super.__init__ (that instantiates configuration)

        :param default_input_value:
        :param params:
        :param name:
        :param prefs:
        :param context:
        """
        # MODIFIED 9/20/16 NEW:  replaced above with None
        processes = processes or []
        monitored_output_states = monitored_output_states or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]
        # MODIFIED 9/20/16 END

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(processes=processes,
                                                 controller=controller,
                                                 enable_controller=enable_controller,
                                                 monitored_output_states=monitored_output_states,
                                                 params=params)

        self.configuration = NotImplemented
        self.processes = []
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


        # # MODIFIED 7/21/16 OLD:
        # self.controller = self.paramsCurrent[kwController](params={SYSTEM: self})

        # MODIFIED 7/21/16 NEW:
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
            self.value = self.processes[-1][PROCESS].outputState.value

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

        If self.processes is empty, instantiate default Process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        If inputs is specified, check that it's length equals the number of processes
        If inputs is not specified, compose from the input for each Process (value specified or, if None, default)
        Note: specification of input for system takes precedence over specification for processes

        # ??STILL THE CASE, OR MOVED TO instantiate_graph:
        Iterate through Process.mechanismList for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.graph
        """

        self.processes = self.paramsCurrent[kwProcesses]

        # # self.processList = []
        # self.processList = SystemProcessList(self)
        # names = self.processList.processNames
        # processes = self.processList.processes

        self.mechanismsDict = {}
        self.all_mech_tuples = []
        self.allMechanisms = SystemMechanismsList(self)

        # Assign default Process if kwProcess is empty, or invalid
        if not self.processes:
            from PsyNeuLink.Functions.Process import Process_Base
            self.processes.append((Process_Base(), None))

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS
        for i in range(len(self.processes)):

            # Convert all entries to (process, input) tuples, with None as filler for absent inputs
            if not isinstance(self.processes[i], tuple):
                self.processes[i] = (self.processes[i], None)

            if inputs is None:
                # FIX: ASSIGN PROCESS INPUTS TO SYSTEM INPUTS
                process = self.processes[i][PROCESS]
                process_input = []
                for process_input_state in process.processInputStates:
                    process_input.extend(process_input_state.value)
                self.processes[i] = (process, process_input)
            # If input was provided on command line, assign that to input item of tuple
            else:
                # Number of inputs in variable must equal number of self.processes
                if len(inputs) != len(self.processes):
                    raise SystemError("Number of inputs ({0}) must equal number of Processes in kwProcesses ({1})".
                                      format(len(inputs), len(self.processes)))
                # Replace input item in tuple with one from variable
                self.processes[i] = (self.processes[i][PROCESS], inputs[i])
            # Validate input
            if (not self.processes[i][PROCESS_INPUT] is None and
                    not isinstance(self.processes[i][PROCESS_INPUT],(numbers.Number, list, np.ndarray))):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, self.processes[i][PROCESS_INPUT]))

            process = self.processes[i][PROCESS]
            input = self.processes[i][PROCESS_INPUT]

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

            # MODIFIED 9/27/16:
            process.mechanisms = ProcessMechanismsList(process)

        # self.processList = []
        self.processList = SystemProcessList(self)

    def instantiate_graph(self, context=None):
        # DOCUMENTATION: EXPAND BELOW
        """Return acyclic graph of system (ignore feedback projections)
        
        # Mark mechanisms as ORIGIN, TERMINAL and in need of INITALIZATION
        # Prune projections from processes or mechanisms in processes not in the system
        # Ignore feedback projections in construction of dependency_set
        # Assign self (system) to each mechanism.systems with mechanism's status (ORIGIN, TERMINAL, INITIALIZE) as value
        # Construct self.mechanismsList, self.mech_tuples, self.allMechanisms, self.mechanismDict

        """

        # Use to recursively traverse processes
        def build_dependency_sets_by_traversing_projections(sender_mech):

            # Delete any projections to mechanism from processes or mechanisms in processes not in current system
            for input_state in sender_mech.inputStates.values():
                for projection in input_state.receivesFromProjections:
                    sender = projection.sender.owner
                    system_processes = self.processList.processes
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

            # Assign as TERMINAL (or SINGELTON) if it has no outgoing projections and is not a Comparator or
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

                    # Do not include dependency (or receiver on sender) for this projection and end this branch of the
                    #    traversal if the receiver has already been encountered, but do mark for initialization
                    # Note: this is because it is a feedback connection, which introduces a cycle into the graph
                    #       that precludes use of toposort to determine order of execution;
                    #       however, the feedback projection will still be used during execution
                    #       and so should be initialized
                    if receiver_tuple in self.graph:
                        # Try assigning receiver as dependent of current mechanism and test toposort
                        try:
                            # If receiver_tuple already has dependencies in its set, add sender_mech to set
                            if self.graph[receiver_tuple]:
                                self.graph[receiver_tuple].add(self.allMechanisms.get_tuple_for_mech(sender_mech))
                            # If receiver_tuple set is empty, assign sender_mech to set
                            else:
                            # If receiver_tuple set is empty, assign sender_mech to set
                                self.graph[receiver_tuple] = {self.allMechanisms.get_tuple_for_mech(sender_mech)}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.graph))
                        # If making receiver dependent on sender produced a cycle (feedback loop), remove from graph
                        except ValueError:
                            self.graph[receiver_tuple].remove(self.allMechanisms.get_tuple_for_mech(sender_mech))
                            # Assign sender_mech INITIALIZE as system status if not ORIGIN or not yet assigned
                            if not sender_mech.systems or sender_mech.systems[self] != ORIGIN:
                                sender_mech.systems[self] = INITIALIZE
                                continue

                    # Assign receiver as dependent on sender mechanism
                    try:
                        # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                        # FIX: DOES THAT SCREW UP ORIGINS?
                        self.graph[receiver_tuple].add(self.allMechanisms.get_tuple_for_mech(sender_mech))
                    except KeyError:
                        self.graph[receiver_tuple] = {self.allMechanisms.get_tuple_for_mech(sender_mech)}

                    if not sender_mech.systems:
                        sender_mech.systems[self] = INTERNAL

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver)

        self.graph = {}

        for process_tuple in self.processes:

            process = process_tuple[0]
            self.temp_process = process
            # process_mech_list = ProcessMechanismsList(process)
            mech = process.firstMechanism            

            # Treat as ORIGIN if ALL projections to the first mechanism in the process
            #     are from ProcessInputStates belonging to processes in the system
            # Notes:
            # * This precludes a mechanism that is an ORIGIN of a process from being an ORIGIN for the system
            #       if it receives any projections from any other mechanisms in the system (including other processes);
            #       that is, if it receives any projections other than from a ProcessInputState
            # * This does allow a mechanism to be the ORIGIN (but *only* the ORIGIN) for > 1 process in the system
            if any(
                    all(
                        isinstance(projection.sender, ProcessInputState) and
                                        projection.sender.owner in self.processList.processes
                        for projection in input_state.receivesFromProjections)
                    for input_state in mech.inputStates.values()):
                # assign its set value as empty, marking it as a "leaf" in the graph
                mech_tuple = self.allMechanisms.get_tuple_for_mech(mech)
                self.graph[mech_tuple] = set()
                mech.systems[self] = ORIGIN

            build_dependency_sets_by_traversing_projections(mech)

        #print graph
        # if self.verbosePref:
        for receiver_mech_tuple, dep_set in self.graph.items():
            name = receiver_mech_tuple[MECHANISM].name
            if not dep_set:
                print("{} is {}".format(name, receiver_mech_tuple[MECHANISM].systems[self]))
                print("{} has no dependents".format(name))
            else:
                print("{} is {}".format(name, receiver_mech_tuple[MECHANISM].systems[self]))
                print("Senders to {}:".format(name))
                for sender_mech_tuple in dep_set:
                    print("\t{}".format(sender_mech_tuple[MECHANISM].name))

        self.origin_mech_tuples = []
        self.terminal_mech_tuples = []

        # For each mechanism (represented by its tuple) in the graph, add entry to relevant list(s)
        for mech_tuple in self.graph:
            if mech_tuple[MECHANISM].systems[self] in {ORIGIN, SINGLETON}:
                self.origin_mech_tuples.append(mech_tuple)
            if mech_tuple[MECHANISM].systems[self] in {TERMINAL, SINGLETON}:
                self.terminal_mech_tuples.append(mech_tuple)

        # FIX: ASSIGN system.mechanismList = MechanismList(system) and implement get_tuple_for_mech for system
        self.allMechanisms = SystemMechanismsList(self)
        self.originMechanisms = OriginMechanismsList(self)
        self.terminalMechanisms = TerminalMechanismsList(self)

        try:
            self.execution_sets = list(toposort(self.graph))
        except ValueError as e:
            if 'Cyclic dependencies exist' in e.args[0]:
                # if self.verbosePref:
                # print('{} has feedback connections; be sure that the following items are properly initialized:'.
                #       format(self.name))
                raise SystemError("PROGRAM ERROR: cycle (feedback loop) in {} not detected by instantiate_graph ".
                                  format(self.name))

        # Create instance of sequential (execution) list:
        self.execution_list = toposort_flatten(self.graph, sort=False)

    def identify_origin_and_terminal_mechanisms(self):
        """Find origin and terminal Mechanisms of graph and assign to self.originMechanisms and self.terminalMechanisms

        Identify origin and terminal Mechanisms in graph:
            - origin mechanisms are ones that do not receive projections from any other mechanisms in the System
            - terminal mechanisms are ones that do not send projections to any other mechanisms in the System
        Assigns the (Mechanism, runtime_params, phase) tuple for each to
            self.origin_mech_tuples and self.terminal_mech_tuples lists
        Instantiates self.originMechanisms and self.terminalMechanisms attributes
            these are convenience lists that refers to the Mechanism item of each tuple
            in self.origin_mech_tuples and self.terminal_mech_tuples lists
        """

        # Get mech_tuples for all mechanisms in the graph
        # Notes
        # * each key in the graph dict is a mech_tuple (Mechanisms, runtime_param, phase) for a receiver;
        # * each entry is a set of mech_tuples that send to the receiver
        # * every mechanism in the System appears in the graph as a receiver, even if its entry (set of senders) is null
        # *    therefore, list of all keys == list of all mechanisms == list of all receivers
        receiver_mech_tuples = set(list(self.graph.keys()))

        set_of_sender_mech_tuples = set()
        self.origin_mech_tuples = []
        # For each mechanism (represented by its tuple) in the graph
        for receiver in self.graph:
            # Add entry (i.e., its set of senders) to the sender set
            set_of_sender_mech_tuples = set_of_sender_mech_tuples.union(self.graph[receiver])
            # If entry is null (i.e., mechanism has no senders), add it to list of origin mechanism tuples
            if not self.graph[receiver]:
                self.origin_mech_tuples.append(receiver)
        # Sort by phase
        self.origin_mech_tuples.sort(key=lambda mech_tuple: mech_tuple[PHASE_SPEC])

        # FIX: THIS ELIMINATES MECHANISMS THAT ARE A TERMINAL IN ONE PROCESS BUT A SENDER (INCLUDING ORIGIN) IN ANOTHER
        # FIX: NOTE:  IT'S STATUS AS A RECEIVER IS SUPPRESSED ABOVE IN ORDER TO AVOID A CYCLIC GRAPH
        # FIX:        HOWEVER CAN FIND OUT WHETHER IT IS A TERMINAL BY EXAMINING PROJECTIONS??
        # FIX:        OR FLAG ABOVE USING ATTRIBUTES?
        # Terminal mechanisms are those in receiver (full) set that are NOT themselves senders (i.e., in the sender set)
        self.terminal_mech_tuples = list(receiver_mech_tuples - set_of_sender_mech_tuples)
        # Sort by phase
        self.terminal_mech_tuples.sort(key=lambda mech_tuple: mech_tuple[PHASE_SPEC])

        # FIX: ELIMINATE OR REWORK BASED ON REFACTORING OF instantiate_graph ABOVE:
        # Instantiate lists of origin and terimal mechanisms,
        #    and assign the mechanism's status in the system to its entry in the mechanism's systems dict
        self.originMechanisms = OriginMechanismsList(self)
        for mech in self.originMechanisms:
            mech.systems[self] = ORIGIN
        self.terminalMechanisms = TerminalMechanismsList(self)
        for mech in self.originMechanisms:
            mech.systems[self] = TERMINAL

# FIX: MAY NEED TO ASSIGN OWNERSHIP OF MECHANISMS IN PROCESSES TO THEIR PROCESSES (OR AT LEAST THE FIRST ONE)
# FIX: SO THAT INPUT CAN BE ASSIGNED TO CORRECT FIRST MECHANISMS (SET IN GRAPH DOES NOT KEEP TRACK OF ORDER)
    def assign_output_states(self):
        """Assign outputStates for System (the values of which will comprise System.value)

        Note:
        * Current implementation simply assigns terminal mechanisms as outputStates
        * This method is included so that sublcasses and/or future versions can override it to make custom assignments

        """
        for mech in self.terminalMechanisms.mechanisms:
            self.outputStates[mech.name] = mech.outputStates

    def execute(self,
                inputs=None,
                time_scale=NotImplemented,
                context=None
                ):
# DOCUMENT: NEEDED -- INCLUDED HANDLING OF phaseSpec
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
            report_process_output = any(process[0].reportOutputPref for process in self.processes)

        if time_scale is NotImplemented:
            self.timeScale = TimeScale.TRIAL

        #region ASSIGN INPUTS TO PROCESSES
        # Assign each item of input to the value of a Process.input_state which, in turn, will be used as
        #    the input to the mapping projection to the first (origin) Mechanism in that Process' configuration
        if inputs is None:
            pass
        else:
            if len(inputs) != len(list(self.originMechanisms)):
                raise SystemError("Number of inputs ({0}) to {1} does not match its number of origin Mechanisms ({2})".
                                  format(len(inputs), self.name,  len(list(self.originMechanisms)) ))
            for i in range(len(inputs)):
                input = inputs[i]
                process = self.processes[i][PROCESS]

                # Make sure there is an input, and if so convert it to 2D np.ndarray (required by Process
                if not input or input is NotImplemented:
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
                    if status is ORIGIN and process.reportOutputPref:
                        process.report_process_initiation()

            # Only update Mechanism on time_step(s) determined by its phaseSpec (specified in Mechanism's Process entry)
# FIX: NEED TO IMPLEMENT FRACTIONAL UPDATES (IN Mechanism.update()) FOR phaseSpec VALUES THAT HAVE A DECIMAL COMPONENT
            if phase_spec == (CentralClock.time_step % (self.phaseSpecMax +1)):
                # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
                mechanism.execute(time_scale=self.timeScale,
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
                self.variable = self.variable * 0
            i += 1
        #endregion


        # region EXECUTE LEARNING FOR EACH PROCESS

        for process_tuple in self.processes:
            process = process_tuple[PROCESS]
            if process.learning and process.learning_enabled:
                process.execute_learning(context=context)
        # endregion


        #region EXECUTE CONTROLLER

# FIX: 1) RETRY APPENDING TO EXECUTE LIST AND COMPARING TO THIS VERSION
# FIX: 2) REASSIGN INPUT TO SYSTEM FROM ONE DESIGNATED FOR EVC SIMULUS (E.G., StimulusPrediction)

        # Only call controller if this is not a controller simulation run (to avoid infinite recursion)
        if not kwEVCSimulation in context and self.enable_controller:
            try:
                if self.controller.phaseSpec == (CentralClock.time_step % (self.phaseSpecMax +1)):
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
        return list(m[MECHANISM] for m in self.graph)

    def mech_names_in_graph(self):
        return list(m[MECHANISM].name for m in self.graph)

    def mech_status_in_graph(self):
        return list(m[MECHANISM].systems for m in self.graph)

    def report_system_initiation(self):

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        if CentralClock.time_step == 0:
            print("\n\'{}\'{} executing with: **** (time_step {}) ".
                  format(self.name, system_string, CentralClock.time_step))
            processes = list(process[0].name for process in self.processes)
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
            if mech[MECHANISM].phaseSpec == (CentralClock.time_step % (self.phaseSpecMax + 1)):
                print("- output for {0}: {1}".format(mech[MECHANISM].name,
                                                     re.sub('[\[,\],\n]','',str(mech[MECHANISM].outputState.value))))


    class InspectOptions(AutoNumber):
        ALL = ()
        EXECUTION_SETS = ()
        EXECUTION_LIST = ()
        ORIGIN_MECHANISMS = ()
        TERMINAL_MECHANISMS = ()
        ALL_OUTPUTS = ()
        ALL_OUTPUT_LABELS = ()
        PRIMARY_OUTPUTS = ()
        PRIMARY_OUTPUT_LABELS = ()
        MONITORED_OUTPUTS = ()
        MONITORED_OUTPUT_LABELS = ()
        FLAT_OUTPUT = ()
        DICT_OUTPUT = ()


    def inspect(self, options=None):
        """Print execution_sets, execution_list, origin and terminal mechanisms, outputs, output labels
        """

        # # IMPLEMENTATION NOTE:  Stub for implementing options
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
