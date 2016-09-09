#
# *********************************************  Process ***************************************************************
#

import re
import math
import PsyNeuLink.Functions
from PsyNeuLink.Functions.ShellClasses import *
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Functions.Mechanisms.Mechanism import Mechanism_Base, mechanism, is_mechanism_spec
from PsyNeuLink.Functions.Projections.Projection import is_projection_spec, add_projection_to
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.States.State import instantiate_state_list, instantiate_state
from PsyNeuLink.Functions.States.ParameterState import ParameterState


# *****************************************    PROCESS CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# Labels for items in configuration entry tuples
OBJECT = 0
PARAMS = 1
PHASE = 2

# FIX: NOT WORKING WHEN ACCESSED AS DEFAULT:
DEFAULT_PROJECTION_MATRIX = AUTO_ASSIGN_MATRIX
# DEFAULT_PROJECTION_MATRIX = IDENTITY_MATRIX

ProcessRegistry = {}


class ProcessError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


# Process factory method:
def process(default_input_value=NotImplemented,
            process_spec=NotImplemented,
            configuration=[Mechanism_Base.defaultMechanism],
            default_projection_matrix=DEFAULT_PROJECTION_MATRIX,
            learning=None,
            params=None,
            name=NotImplemented,
            prefs=NotImplemented,
            context=NotImplemented):
    """Return subclass specified by process_spec or default process

    If called with no arguments or first argument is NotImplemented,  instantiates process with
        subclass Mechanism (currently DDM)
    If called with a name string, uses it as the name for an instantiation of the Process
    If a params dictionary is included, it is passed to the Process (inclulding kwConfig)

    :param process_spec: (Process_Base, str or specification dict)
    :param params: (dict)
    :param context: (str)
    :return: (Process object or None)
    """

    # Called with descriptor keyword
    if process_spec in ProcessRegistry:
        return ProcessRegistry[process_spec].processSubclass(params=params, context=context)

    # Called with a string that is not in the Registry, so return default type with the name specified by the string
    elif isinstance(process_spec, str):
        return Process_Base(name=process_spec, params=params, context=context)

    # Called with Mechanism specification dict (with type and params as entries within it), so:
    #    - get mech_type from kwMechanismType entry in dict
    #    - pass all other entries as params
    elif isinstance(process_spec, dict):
        # Get Mechanism type from kwMechanismType entry of specification dict
        return Process_Base(context=context, **process_spec)

    # Called without a specification, so return Process with default mechanism
    elif process_spec is NotImplemented:
        return Process_Base(default_input_value=default_input_value,
                            configuration=configuration,
                            default_projection_matrix=default_projection_matrix,
                            learning=learning,
                            params=params,
                            name=name,
                            prefs=prefs,
                            context=context)

    # Can't be anything else, so return empty
    else:
        return None


kwProcessInputState = 'ProcessInputState'
from PsyNeuLink.Functions.States.OutputState import OutputState

# DOCUMENT:  HOW DO MULTIPLE PROCESS INPUTS RELATE TO # OF INPUTSTATES IN FIRST MECHANISM
#            WHAT HAPPENS IF LENGTH OF INPUT TO PROCESS DOESN'T MATCH LENGTH OF VARIABLE FOR FIRST MECHANISM??


class Process_Base(Process):
# DOCUMENT:  CONFIGURATION FORMAT:  (Mechanism <, PhaseSpec>) <, Projection,> (Mechanism <, PhaseSpec>)
# DOCUMENT:  Projections SPECIFIED IN A CONFIGURATION MUST BE Mapping Projections
# DOCUMENT:  PhaseSpec:
#   - phaseSpec for each Mechanism in Process::
#        integers:
#            specify time_step (phase) on which mechanism is updated (when modulo time_step == 0)
#                - mechanism is fully updated on each such cycle
#                - full cycle of System is largest phaseSpec value
#        floats:
#            values to the left of the decimal point specify the "cascade rate":
#                the fraction of the outputvalue used as the input to any projections on each (and every) time_step
#            values to the right of the decimal point specify the time_step (phase) at which updating begins
# DOCUMENT: UPDATE CLASS AND INSTANCE METHODS LISTS/DESCRIPTIONS
    """Implement abstract class for Process category of Function class

    Description:
        A Process is defined by a CONFIGURATION param (list of Mechanisms) and a time scale.  Executing a Process
         executes the Mechanisms in the order that they appear in the configuration.

    Instantiation:
        A Process can be instantiated in one of two ways:
            - by a direct call to Process()
            [TBI: - in a list to a call to System()]
        A Process instantiates its configuration by assigning:
            - a projection from the Process to the inputState of the first mechanism in the configuration
            - a projection from each mechanism in the list to the next (if one is not already specified)
                if the length of the preceding mechanism's outputState == the length of the follower's inputState
                    the identity matrix is used
                if the lengths are not equal, the unit full connectivity matrix is used
                if kwLearning has been specified:
                    the unit full connectivity matrix is used
                    kwLearning is specified for all Projections that have not otherwise been specified
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
# DOCUMENT:  UPDATE TO INCLUDE Mechanism, Projection, Mechanism FORMAT, AND (Mechanism, Cycle) TUPLE
            CONFIGURATION (list): (default: single Mechanism_Base.defaultMechanism)
                Each config_entry must be one of the following, that is used to instantiate the mechanisms in the list:
                    + Mechanism object
                    + Mechanism type (class) (e.g., DDM)
                    + descriptor keyword for a Mechanism type (e.g., kwDDM)
                    + specification dict for Mechanism; the dict can have the following entries (see Mechanism):
                        + kwMechanismType (Mechanism subclass): if absent, Mechanism_Base.defaultMechanism is used
                        + entries with keys = standard args of Mechanism.__init__:
                            "input_template":<value>
                            FUNCTION_PARAMS:<dict>
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
                                + kwInputStateParams:<dict>
                                + kwParameterStateParams:<dict>
                           [TBI + kwOutputStateParams:<dict>]
                                - each dict will be passed to the corresponding State
                                - params can be any permissible executeParamSpecs for the corresponding State
                                - dicts can contain the following embedded dicts:
                                    + FUNCTION_PARAMS:<dict>:
                                         will be passed the State's execute method,
                                             overriding its paramInstanceDefaults for that call
                                    + kwProjectionParams:<dict>:
                                         entry will be passed to all of the State's projections, and used by
                                         by their execute methods, overriding their paramInstanceDefaults for that call
                                    + kwMappingParams:<dict>:
                                         entry will be passed to all of the State's Mapping projections,
                                         along with any in a kwProjectionParams dict, and override paramInstanceDefaults
                                    + kwControlSignalParams:<dict>:
                                         entry will be passed to all of the State's ControlSignal projections,
                                         along with any in a kwProjectionParams dict, and override paramInstanceDefaults
                                    + <projectionName>:<dict>:
                                         entry will be passed to the State's projection with the key's name,
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
        + paramClassDefaults = {CONFIGURATION: [Mechanism_Base.defaultMechanism],
                                kwTimeScale: TimeScale.TRIAL}

    Class methods:
        - execute(input, control_signal_allocations, time_scale):
            executes the process by calling execute_functions of the mechanisms (in order) in the configuration list
            assigns input to sender.output (and passed through mapping) of first mechanism in the configuration list
            assigns output of last mechanism in the configuration list to self.output
            returns output after either one time_step or the full trial (determined by time_scale)
        - get_configuration(): returns configuration (list)
        - get_mechanism_dict(): returns mechanismDict (dict)
        - register_process(): registers process with ProcessRegistry
        [TBI: - adjust(control_signal_allocations=NotImplemented):
            modifies the control_signal_allocations while the process is executing;
            calling it without control_signal_allocations functions like interrogate
            returns (responseState, accuracy)
        [TBI: - interrogate(): returns (responseState, accuracy)
        [TBI: - terminate(): terminates the process and returns output
        [TBI: - accuracy(target):
            a function that uses target together with the configuration's output.value(s)
            and its accuracyFunction to return an accuracy measure;
            the target must be in a configuration-appropriate format (checked with call)

    Instance attributes:
# DOCUMENT:  EXPLAIN (Mechanism <, Cycle>) <, Projection,> (Mechanism <, Cycle>) FORMAT
        + configuration (list): set in params[CONFIGURATION]
            an ordered list of tuples that defines how the process is carried out;
                (default: the default mechanism for the Mechanism class -- currently: DDM)
                Note:  this is constructed from the CONFIGURATION param, which may or may not contain tuples;
                       all entries of CONFIGURATION param are converted to tuples for self.configuration
                       for entries that are not tuples, None is used for the param (2nd) item of the tuple
# DOCUMENT: THESE HAVE BEEN REPLACED BY processInputStates (BELOW)
        # + sendsToProjections (list)           | these are used to instantiate a projection from Process
        # + owner (None)               | to first mechanism in the configuration list
        # + value (value)                       | value is used to specify input to Process;
        #                                       | it is zeroed after executing the first item in the configuration
        + processInputStates (OutputState:
            instantiates projection(s) from Process to first Mechanism in the configuration
        + outputState (MechanismsState object) - reference to OutputState of last mechanism in configuration
            updated with output of process each time process.execute is called
        + phaseSpecMax (int) - integer component of maximum phaseSpec for Mechanisms in configuration
        + system (System) - System to which Process belongs
        + timeScale (TimeScale): set in params[kwTimeScale]
             defines the temporal "granularity" of the process; must be of type TimeScale
                (default: TimeScale.TRIAL)
        + mechanismDict (dict) - dict of mechanisms used in configuration (one config_entry per mechanism type):
            - key: mechanismName
            - value: mechanism
        + mechanismList (list) - list of (Mechanism, params, phase_spec) tuples in order specified in configuration
        + mechanismNames (list) - list of mechanism names in mechanismList
        + monitoringMechanismList (list) - list of (MonitoringMechanism, params, phase_spec) tuples derived from
                                           MonitoringMechanisms associated with any LearningSignals
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
    paramClassDefaults.update({kwTimeScale: TimeScale.TRIAL})

    default_configuration = [Mechanism_Base.defaultMechanism]

    def __init__(self,
                 default_input_value=NotImplemented,
                 configuration=default_configuration,
                 default_projection_matrix=DEFAULT_PROJECTION_MATRIX,
                 learning=None,
                 params=None,
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

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(configuration=configuration,
                                                 default_projection_matrix=default_projection_matrix,
                                                 learning=learning,
                                                 params=params)

        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name
        self.functionName = self.functionType

        self.configuration = NotImplemented
        self.mechanismDict = {}
        self.processInputStates = []
        self.phaseSpecMax = 0
        self.function = self.execute

        register_category(self, Process_Base, ProcessRegistry, context=context)

        if context is NotImplemented:
            # context = self.__class__.__name__
            context = kwInit + self.name + kwSeparator + kwProcessInit

        super(Process_Base, self).__init__(variable_default=default_input_value,
                                           param_defaults=params,
                                           name=self.name,
                                           prefs=prefs,
                                           context=context)
        if self.prefs.reportOutputPref:
            print("\n{0} initialized with:\n- configuration: [{1}]".
                  format(self.name, self.mechanismNames.__str__().strip("[]")))

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

    def instantiate_attributes_before_function(self, context=NotImplemented):
        """Call methods that must be run before function method is instantiated

        Need to do this before instantiate_function as mechanisms in configuration must be instantiated
            in order to assign input projection and self.outputState to first and last mechanisms, respectively

        :param context:
        :return:
        """
        self.instantiate_configuration(context=context)
        # super(Process_Base, self).instantiate_function(context=context)

    def instantiate_function(self, context=NotImplemented):
        """Override Function.instantiate_function:

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a Process object)
        - suppress validation (and attendant execution) of Process execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in the configuration have already been validated

        :param context:
        :return:
        """

        if self.paramsCurrent[FUNCTION] != self.execute:
            print("Process object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[FUNCTION], FUNCTION)
            self.paramsCurrent[FUNCTION] = self.execute
        # If validation pref is set, instantiate and execute the Process
        if self.prefs.paramValidationPref:
            super(Process_Base, self).instantiate_function(context=context)
        # Otherwise, just set Process output info to the corresponding info for the last mechanism in the configuration
        else:
            self.value = self.configuration[-1][OBJECT].outputState.value

# DOCUMENTATION:
#         Uses paramClassDefaults[CONFIGURATION] == [Mechanism_Base.defaultMechanism] as default
#         1) ITERATE THROUGH CONFIG LIST TO PARSE AND INSTANTIATE EACH MECHANISM ITEM
#             - RAISE EXCEPTION IF TWO PROJECTIONS IN A ROW
#         2) ITERATE THROUGH CONFIG LIST AND ASSIGN PROJECTIONS (NOW THAT ALL MECHANISMS ARE INSTANTIATED)
#
#
# FIX:
#     ** PROBLEM: self.value IS ASSIGNED TO variableInstanceDefault WHICH IS 2D ARRAY,
        # BUT PROJECTION EXECUTION FUNCTION TAKES 1D ARRAY2222
#         Assign projection from Process (self.value) to inputState of the first mechanism in the configuration
#     **?? WHY DO THIS, IF SELF.VALUE HAS BEEN ASSIGNED AN INPUT VALUE, AND PROJECTION IS PROVIDING INPUT TO MECHANISM??
#         Assigns variableInstanceDefault to variableInstanceDefault of first mechanism in configuration

    def instantiate_configuration(self, context):
        # DOCUMENT:  Projections SPECIFIED IN A CONFIGURATION MUST BE A Mapping Projection
        # DOCUMENT:
        # Each item in Configuration can be a Mechanism or Projection object, class ref, or specification dict,
        #     str as name for a default Mechanism,
        #     keyword (IDENTITY_MATRIX or FULL_CONNECTIVITY_MATRIX) as specification for a default Projection,
        #     or a tuple with any of the above as the first item and a param dict as the second
        """Construct configuration list of Mechanisms and Projections used to execute process

        Iterate through Configuration, parsing and instantiating each Mechanism item;
            - raise exception if two Projections are found in a row;
            - add each Mechanism to mechanismDict and to list of names
            - for last Mechanism in Configuration, assign ouputState to Process.outputState
        Iterate through Configuration, assigning Projections to Mechanisms:
            - first Mechanism in Configuration:
                if it does NOT already have any projections:
                    assign projection(s) from ProcessInputState(s) to corresponding Mechanism.inputState(s):
                if it DOES already has a projection, and it is from:
                    (A) the current Process input, leave intact
                    (B) another Process input, if verbose warn
                    (C) another mechanism in the current process, if verbose warn about recurrence
                    (D) a mechanism not in the current Process or System, if verbose warn
                    (E) another mechanism in the current System, OK so ignore
                    (F) from something other than a mechanism in the system, so warn (irrespective of verbose)
                    (G) a Process in something other than a System, so warn (irrespective of verbose)
            - subsequent Mechanisms:
                assign projections from each Mechanism to the next one in the list:
                - if Projection is explicitly specified as item between them in the list, use that;
                - if Projection is NOT explicitly specified,
                    but the next Mechanism already has a projection from the previous one, use that;
                - otherwise, instantiate a default Mapping projection from previous mechanism to next:
                    use kwIdentity (identity matrix) if len(sender.value == len(receiver.variable)
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if the lengths are not equal
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if kwLearning has been set

        :param context:
        :return:
        """

        configuration = self.paramsCurrent[CONFIGURATION]
        self.mechanismList = []
        self.mechanismNames = []
        self.monitoringMechanismList = []

# FIX: LENGTHEN TUPLE INSTANTIATION HERE (LEN = 3) TO ACCOMODATE phaseSpec, AND ADD TO PARSE BELOW;
# FIX:  DEFAULT: 1 (UPDATE FULLY EVERY CYCLE)
# IMPLEMENTATION NOTE:  for projections, 2nd and 3rd items of tuple are ignored

        #region STANDARDIZE ENTRY FORMAT -------------------------------------------------------------------------------

# FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process.validate_params
        # Convert all entries to (item, params, phaseSpec) tuples, padded with None for absent params and/or phaseSpec
        for i in range(len(configuration)):
            config_item = configuration[i]
            if isinstance(config_item, tuple):
                # If the tuple has only one item, check that it is a Mechanism or Projection specification
                if len(config_item) is 1:
                    if is_mechanism_spec(config_item[0]) or is_projection_spec(config_item[0]):
                        # Pad with None
                        configuration[i] = (config_item[0], None, None)
                    else:
                        raise ProcessError("Item of tuple ({0}) in entry {1} of configuration for {2}"
                                           " is neither a mechanism nor a projection specification".
                                           format(config_item[1], i, self.name))
                # If the tuple has two items, check whether second item is a params dict or a phaseSpec
                #    and assign it to the appropriate position in the tuple, padding other with None
                if len(config_item) is 2:
                    # Mechanism
                    if is_mechanism_spec(config_item[0]):
                        if isinstance(config_item[1], dict):
                            configuration[i] = (config_item[0], config_item[1], None)
                        elif isinstance(config_item[1], (int, float)):
                            configuration[i] = (config_item[0], None, config_item[1])
                        else:
                            raise ProcessError("Second item of tuple ((0}) in item {1} of configuration for {2}"
                                               " is neither a params dict nor phaseSpec (int or float)".
                                               format(config_item[1], i, self.name))
                    # Projection
                    elif is_projection_spec(config_item[0]):
                        if is_projection_spec(config_item[1]):
                            configuration[i] = (config_item[0], config_item[1], None)
                        else:
                            raise ProcessError("Second item of tuple ({0}) in item {1} of configuration for {2}"
                                               " should be 'LearningSignal' or absent".
                                               format(config_item[1], i, self.name))
                    else:
                        raise ProcessError("First item of tuple ((0}) in item {1} of configuration for {2}"
                                           " is neither a mechanism nor a projection spec".
                                           format(config_item[1], i, self.name))
                if len(config_item) > 3:
                    raise ProcessError("The tuple for item {0} of configuration for {1} has more than three items {2}".
                                       format(i, self.name, config_item))
            else:
                # Convert item to tuple, padded with None
                if is_mechanism_spec(configuration[i]) or is_projection_spec(configuration[i]):
                    # Pad with None
                    configuration[i] = (configuration[i], None, None)
                else:
                    raise ProcessError("Item of {1} of configuration for {2}"
                                       " is neither a mechanism nor a projection specification".
                                       format(i, self.name))
        #endregion

        #region VALIDATE CONFIGURATION THEN PARSE AND INSTANTIATE MECHANISM ENTRIES  ------------------------------------

        # - make sure first entry is not a Projection
        # - make sure Projection entries do NOT occur back-to-back (i.e., no two in a row)
        # - instantiate Mechanism entries

        previous_item_was_projection = False
        from PsyNeuLink.Functions.Projections.Projection import Projection_Base

        for i in range(len(configuration)):
            item, params, phase_spec = configuration[i]

            # Get max phaseSpec for Mechanisms in configuration
            if not phase_spec:
                phase_spec = 0
            self.phaseSpecMax = int(max(math.floor(float(phase_spec)), self.phaseSpecMax))

            #region VALIDATE PLACEMENT OF PROJECTION ENTRIES  ----------------------------------------------------------

            # Can't be first entry, and can never have two in a row

            # Config entry is a Projection
            if is_projection_spec(item):
                # Projection not allowed as first entry
                if i==0:
                    raise ProcessError("Projection cannot be first entry in configuration ({0})".format(self.name))
                # Projections not allowed back-to-back
                if previous_item_was_projection:
                    raise ProcessError("Illegal sequence of two adjacent projections ({0}:{1} and {1}:{2})"
                                       " in configuration for {3}".
                                       format(i-1, configuration[i-1], i, configuration[i], self.name))
                previous_item_was_projection = True
                continue

            previous_item_was_projection = False
            mech = item
            #endregion

            #region INSTANTIATE MECHANISM  -----------------------------------------------------------------------------

            # Must do this before assigning projections (below)
            # Mechanism entry must be a Mechanism object, class, specification dict, str, or (Mechanism, params) tuple
            # Don't use params item of tuple (if present) to instantiate Mechanism, as they are runtime only params

            # Entry is NOT already a Mechanism object
            if not isinstance(mech, Mechanism):
                # Note: need full pathname for mechanism factory method, as "mechanism" is used as local variable below
                mech = PsyNeuLink.Functions.Mechanisms.Mechanism.mechanism(mech, context=context)
                if not mech:
                    raise ProcessError("Entry {0} ({1}) is not a recognized form of Mechanism specification".
                                       format(i, mech))
                # Params in mech tuple must be a dict or None
                if params and not isinstance(params, dict):
                    raise ProcessError("Params entry ({0}) of tuple in item {1} of configuration for {2} is not a dict".
                                          format(params, i, self.name))
                # Replace Configuration entry with new tuple containing instantiated Mechanism object and params
                configuration[i] = (mech, params, phase_spec)

            # Entry IS already a Mechanism object
            # Add entry to mechanismList and name to mechanismNames list
            mech.phaseSpec = phase_spec
            # Add Process to the mechanism's list of processes to which it belongs
            if not self in mech.processes:
                mech.processes[self] = INTERNAL
            self.mechanismList.append(configuration[i])
            self.mechanismNames.append(mech.name)
            #endregion
        #endregion

        # Identify origin and terminal mechanisms in the process and
        #    and assign the mechanism's status in the process to its entry in the mechanism's processes dict
        self.firstMechanism = configuration[0][OBJECT]
        self.firstMechanism.processes[self] = ORIGIN
        self.lastMechanism = configuration[-1][OBJECT]
        self.lastMechanism.processes[self] = TERMINAL

        # Assign process outputState to last mechanisms in configuration
        self.outputState = self.lastMechanism.outputState

        # ASSIGN DEFAULT PROJECTION PARAMS
        # If learning is specified for the Process, add to default projection params
        if self.learning:
            # FIX: IF self.learning IS AN ACTUAL LearningSignal OBJECT, NEED TO RESPECIFY AS CLASS + PARAMS
            #      OR CAN THE SAME LearningSignal OBJECT BE SHARED BY MULTIPLE PROJECTIONS?
            #      DOES IT HAVE ANY INTERNAL STATE VARIABLES OR PARAMS THAT NEED TO BE PROJECTIONS-SPECIFIC?
            matrix_spec = (self.default_projection_matrix, self.learning)
        else:
            matrix_spec = self.default_projection_matrix
        projection_params = {FUNCTION_PARAMS:
                                 {MATRIX: matrix_spec}}
        # MODIFIED 9/5/16 END

        #region PARSE, INSTANTIATE AND ASSIGN PROJECTION ENTRIES -------------------------------------------------------
        self.parse_instantiate_assign_projection_entries(configuration=configuration, context=context)

        #endregion
        self.configuration = configuration

        self.instantiate_deferred_inits(context=context)

    def parse_instantiate_assign_projection_entries(self, configuration, context=NotImplemented):

        for i in range(len(configuration)):
                item, params, phase_spec = configuration[i]

                #region FIRST ENTRY

                # Must be a Mechanism (enforced above)
                # Assign input(s) from Process to it if it doesn't already have any
                if i == 0:
                    # Relabel for clarity
                    mechanism = item

                    # Check if first Mechanism already has any projections and, if so, issue appropriate warning
                    if mechanism.inputState.receivesFromProjections:
                        self.issue_warning_about_existing_projections(mechanism, context)

                    # Assign input projection from Process
                    self.assign_process_input_projections(mechanism, context=context)
                    continue
                #endregion

                #region SUBSEQUENT ENTRIES

                # Item is a Mechanism
                if isinstance(item, Mechanism):

                    preceding_item = configuration[i-1][OBJECT]

                    # PRECEDING ITEM IS A PROJECTION
                    # # MODIFIED 9/8/16 OLD:
                    # # If preceding entry was a projection no need to do anything
                    # #    (as the current Mechanism should have already been assigned as the receiver)
                    # if isinstance(preceding_item, Projection):
                    #     continue
                    # MODIFIED 9/8/16 NEW:
                    # If preceding entry was a projection no need create one,
                    #    however, if self.learning arg is specified, need to check whether projection has a learningSignal
                    #    and, if not, assign the one specified
                    from PsyNeuLink.Functions.Projections.LearningSignal import LearningSignal
                    if isinstance(preceding_item, Projection):
                        if self.learning:
                            # try:
                            #     matrix_param_state = preceding_item.parameterStates[MATRIX]
                            # except KeyError:
                            #     # FIX: ADD MATRIX PARAMETER STATE TO preceding_item.parameterStates
                            # # ITERATE THROUGH THE FOLLOWING TO CHECK IF ANY HAVE A LearningSignal SPECIFICATION
                            #     # FIX: ADD LEARNING SIGNAL.copy TO matrix_param_state

                            # Check if preceding_item has a matrix parameterState and, if so, if it has a learningSignal
                            try:
                                has_learning_signal = False
                                if (any(isinstance(projection, LearningSignal) for
                                        projection in preceding_item.parameterStates[MATRIX].receivesFromProjections)):
                                    has_learning_signal = True

                            # preceding_item doesn't have a parameterStates attrib, so assign one with self.learning
                            except AttributeError:
                                # Instantiate parameterStates Ordered dict with ParameterState for self.learning
                                preceding_item.parameterStates = instantiate_state_list(
                                                                                    owner=preceding_item,
                                                                                    state_list=[(MATRIX,
                                                                                                 self.learning)],
                                                                                    state_type=ParameterState,
                                                                                    state_param_identifier=kwParameterState,
                                                                                    constraint_value=self.learning,
                                                                                    constraint_value_name=LEARNING_SIGNAL,
                                                                                    context=context)

                            # preceding_item has parameterStates but not (yet!) one for MATRIX, so instantiate it
                            except KeyError:
                                # Instantiate ParameterState for MATRIX
                                preceding_item.parameterStates[MATRIX] = instantiate_state(
                                                                                    owner=preceding_item,
                                                                                    state_type=ParameterState,
                                                                                    state_name=MATRIX,
                                                                                    state_spec=kwParameterState,
                                                                                    state_params=self.learning,
                                                                                    constraint_value=self.learning,
                                                                                    constraint_value_name=LEARNING_SIGNAL,
                                                                                    context=context)

                            # preceding_item has parameterState for MATRIX,
                            else:
                                if has_learning_signal:
                                    # MODIFIED 8/13/16:
                                    # FIX: ?? SHOULD THIS USE assign_defaults:
                                    # Update matrix params with any specified by LearningSignal
                                    preceding_item.parameterStates[MATRIX].paramsCurrent.update(self.learning.user_params)
                                else:
                                    # Add learning signal to projection
                                    add_projection_to(preceding_item,
                                                      preceding_item.parameterStates[MATRIX],
                                                      projection_spec=self.learning)
                        continue
                    # MODIFIED 9/8/16 END


                    # Preceding item was a Mechanism, so check if a Projection needs to be instantiated between them
                    # Check if Mechanism already has a projection from the preceding Mechanism, by testing whether the
                    #    preceding mechanism is the sender of any projections received by the current one's inputState

    # FIX: THIS SHOULD BE DONE FOR ALL INPUTSTATES
    # FIX: POTENTIAL PROBLEM - EVC *CAN* HAVE MULTIPLE PROJECTIONS FROM (DIFFERENT outputStates OF) THE SAME MECHANISM

                    # PRECEDING ITEM IS A MECHANISM
                    projection_list = item.inputState.receivesFromProjections
                    projection_found = False
                    for projection in projection_list:
                        # Current mechanism DOES receive a projection from the preceding item
                        if preceding_item == projection.sender.owner:
                            projection_found = True
                            if self.learning:
                                # Make sure projection includes a learningSignal and add one if it doesn't
                                try:
                                    matrix_param_state = projection.parameterStates['matrix']

                                # projection doesn't have a parameterStates attrib, so assign one with self.learning
                                except AttributeError:
                                    # Instantiate parameterStates Ordered dict with ParameterState for self.learning
                                    projection.parameterStates = instantiate_state_list(
                                                                                    owner=preceding_item,
                                                                                    state_list=[(MATRIX,
                                                                                                 self.learning)],
                                                                                    state_type=ParameterState,
                                                                                    state_param_identifier=kwParameterState,
                                                                                    constraint_value=self.learning,
                                                                                    constraint_value_name=LEARNING_SIGNAL,
                                                                                    context=context)

                                # projection has parameterStates but not (yet!) one for MATRIX,
                                #    so instantiate it with self.learning
                                except KeyError:
                                    # Instantiate ParameterState for MATRIX
                                    projection.parameterStates[MATRIX] = instantiate_state(
                                                                                    owner=preceding_item,
                                                                                    state_type=ParameterState,
                                                                                    state_name=MATRIX,
                                                                                    state_spec=kwParameterState,
                                                                                    state_params=self.learning,
                                                                                    constraint_value=self.learning,
                                                                                    constraint_value_name=LEARNING_SIGNAL,
                                                                                    context=context)

                                # Check if projection's matrix param has a learningSignal
                                else:
                                    if not (any(isinstance(projection, LearningSignal) for
                                                projection in matrix_param_state.receivesFromProjections)):
                                        add_projection_to(projection,
                                                          matrix_param_state,
                                                          projection_spec=self.learning)
                    # MODIFIED 9/8/16 END
                                if self.prefs.verbosePref:
                                    print("LearningSignal added to projection from mechanism {0} to mechanism {1} "
                                          "in configuration of {2}".format(preceding_item.name, mech.name, self.name))

                            break

                    if not projection_found:
                        # No projection found, so instantiate mapping projection from preceding mechanism to current one;
                        # Note:  If self.learning arg is specified, it has already been added to projection_params above
                        Mapping(sender=preceding_item,
                                receiver=item,
                                params=projection_params
                                )
                        if self.prefs.verbosePref:
                            print("Mapping projection added from mechanism {0} to mechanism {1} in configuration of {2}".
                                  format(preceding_item.name, mech.name, self.name))
                    # MODIFIED 9/8/16 NEW:


                # Item is a Projection or specification for one
                else:
                    # Instantiate Projection, assigning mechanism in previous entry as sender and next one as receiver
                    # IMPLEMENTATION NOTE:  FOR NOW:
                    #    - ASSUME THAT PROJECTION SPECIFICATION (IN item) IS ONE OF THE FOLLOWING:
                    #        + Projection object
                    #        + Matrix object
                    # #        +  Matrix keyword (IDENTITY_MATRIX or FULL_CONNECTIVITY_MATRIX)
                    #        +  Matrix keyword (use "is_projection" to validate)
                    #    - params IS IGNORED
    # 9/5/16:
    # FIX: IMPLEMENT validate_params TO VALIDATE PROJECTION SPEC USING Projection.is_projection
    # FIX: ADD SPECIFICATION OF PROJECTION BY KEYWORD:
    # FIX: ADD learningSignal spec if specified at Process level (overrided individual projection spec?)

                    # FIX: PARSE/VALIDATE ALL FORMS OF PROJECTION SPEC (ITEM PART OF TUPLE) HERE:
                    # FIX:                                                          CLASS, OBJECT, DICT, STR, TUPLE??
                    # IMPLEMENT: MOVE State.instantiate_projections_to_state(), check_projection_receiver()
                    #            and parse_projection_ref() all to Projection_Base.__init__() and call that
                    #           VALIDATION OF PROJECTION OBJECT:
                    #                MAKE SURE IT IS A Mapping PROJECTION
                    #                CHECK THAT SENDER IS configuration[i-1][OBJECT]
                    #                CHECK THAT RECEVIER IS configuration[i+1][OBJECT]

                    sender=configuration[i-1][OBJECT]
                    receiver=configuration[i+1][OBJECT]

                    # projection spec is an instance of a Mapping projection
                    if isinstance(item, Mapping):
                        # Check that Projection's sender and receiver are to the mechanism before and after it in the list
                        # IMPLEMENT: CONSIDER ADDING LEARNING TO ITS SPECIFICATION?
    # FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process.validate_params
                        if not item.sender.owner is sender:
                            raise ProcessError("Sender of projection ({}) specified in item {} of configuration for {} "
                                               "is not the mechanism ({}) that proceeds it in the configuration".
                                               format(item.name, i, self.name, sender.name))
                        if not item.receiver.owner is receiver:
                            raise ProcessError("Receiver of projection ({}) specified in item {} of configuration for {} "
                                               "is not the mechanism ({}) that follows it in the configuration".
                                               format(item.name, i, self.name, sender.name))
                        projection = item

                        # MODIFIED 9/6/16 NEW:
                        # TEST
                        if params:
                            projection.matrix = params

                    # # MODIFIED 9/5/16 OLD:
                    # elif ((inspect.isclass(item) and issubclass(item, Mapping)) or
                    #           isinstance(item, np.matrix) or
                    #           (isinstance(item, np.ndarray) and item.ndim == 2) or
                    #           (isinstance(item, str) and (IDENTITY_MATRIX in item or FULL_CONNECTIVITY_MATRIX in item))):
                    #     projection_params = {FUNCTION_PARAMS: {MATRIX: item}}
                    #     projection = Mapping(sender=sender,
                    #                          receiver=receiver,
                    #                          params=projection_params)
                    #     # Reassign Configuration entry
                    #     #    with Projection as OBJECT item and original params as PARAMS item of the tuple
                    #     # IMPLEMENTATION NOTE:  params is currently ignored

                    # MODIFIED 9/5/16 NEW:
                    # projection spec is a Mapping class reference
                    elif inspect.isclass(item) and issubclass(item, Mapping):
                        # MODIFIED 9/6/16 NEW:
                        if params:
                            # Note:  If self.learning arg is specified, it has already been added to projection_params above
                            projection_params = params
                        # MODIFIED 9/6/16 END
                        projection = Mapping(sender=sender,
                                             receiver=receiver,
                                             params=projection_params)

                    # projection spec is a matrix specification, a keyword for one, or a (matrix, LearningSignal) tuple
                    # Note: this is tested above by call to is_projection_spec()
                    elif (isinstance(item, (np.matrix, str, tuple) or
                              (isinstance(item, np.ndarray) and item.ndim == 2))):
                        # # MODIFIED 9/6/16 OLD:
                        # if params:
                        #     projection_params = params
                        # projection = Mapping(sender=sender,
                        #                      receiver=receiver,
                        #                      matrix=item)
                        # MODIFIED 9/6/16 NEW:
                        # If a LearningSignal is explicitly specified for this projection, use it
                        if params:
                            matrix_spec = (item, params)
                        # If a LearningSignal is not specified for this projection but self.learning is specified, use that
                        elif self.learning:
                            matrix_spec = (item, self.learning)
                        # Otherwise, do not include any LearningSignal
                        else:
                            matrix_spec = item
                        projection = Mapping(sender=sender,
                                             receiver=receiver,
                                             matrix=matrix_spec)
                        # MODIFIED 9/6/16 END
                    # MODIFIED 9/5/16 END

                    else:
                        raise ProcessError("Item {0} ({1}) of configuration for {2} is not "
                                           "a valid mechanism or projection specification".format(i, item, self.name))
                    # Reassign Configuration entry
                    #    with Projection as OBJECT item and original params as PARAMS item of the tuple
                    # IMPLEMENTATION NOTE:  params is currently ignored
                    configuration[i] = (projection, params)
                #endregion

    def issue_warning_about_existing_projections(self, mechanism, context=NotImplemented):

        # Check where the projection(s) is/are from and, if verbose pref is set, issue appropriate warnings
        for projection in mechanism.inputState.receivesFromProjections:

            # Projection to first Mechanism in Configuration comes from a Process input
            if isinstance(projection.sender, ProcessInputState):
                # If it is:
                # (A) from self, ignore
                # (B) from another Process, warn if verbose pref is set
                if not projection.sender.owner is self:
                    if self.prefs.verbosePref:
                        print("WARNING: {0} in configuration for {1} already has an input from {2} "
                              "that will be used".
                              format(mechanism.name, self.name, projection.sender.owner.name))

            # (C) Projection to first Mechanism in Configuration comes from one in the Process' mechanismList;
            #     so warn if verbose pref is set
            if projection.sender.owner in self.mechanismList:
                if self.prefs.verbosePref:
                    print("WARNING: first mechanism ({0}) in configuration for {1} receives "
                          "a (recurrent) projection from another mechanism {2} in {1}".
                          format(mechanism.name, self.name, projection.sender.owner.name))

            # Projection to first Mechanism in Configuration comes from a Mechanism not in the Process;
            #    check if Process is in a System, and projection is from another Mechanism in the System
            else:
                try:
                    if (inspect.isclass(context) and issubclass(context, System)):
                        # Relabel for clarity
                        system = context
                    else:
                        system = None
                except:
                    # Process is NOT being implemented as part of a System, so projection is from elsewhere;
                    #  (D)  Issue warning if verbose
                    if self.prefs.verbosePref:
                        print("WARNING: first mechanism ({0}) in configuration for {1} receives a "
                              "projection ({2}) that is not part of {1} or the System it is in".
                              format(mechanism.name, self.name, projection.sender.owner.name))
                else:
                    # Process IS being implemented as part of a System,
                    if system:
                        # (E) Projection is from another Mechanism in the System
                        #    (most likely the last in a previous Process)
                        if mechanism in system.mechanisms:
                            pass
                        # (F) Projection is from something other than a mechanism,
                        #     so warn irrespective of verbose (since can't be a Process input
                        #     which was checked above)
                        else:
                            print("First mechanism ({0}) in configuration for {1}"
                                  " receives a projection {2} that is not in {1} "
                                  "or its System ({3}); it will be ignored and "
                                  "a projection assigned to it by {3}".
                                  format(mechanism.name,
                                         self.name,
                                         projection.sender.owner.name,
                                         context.name))
                    # Process is being implemented in something other than a System
                    #    so warn (irrespecive of verbose)
                    else:
                        print("WARNING:  Process ({0}) being instantiated in context "
                                           "({1}) other than a System ".format(self.name, context))

    def assign_process_input_projections(self, mechanism, context=NotImplemented):
        """Create projection(s) for each item in Process input to inputState(s) of the specified Mechanism

        For each item in Process input:
        - create process_input_state, as sender for Mapping Projection to the mechanism.inputState
        - create the Mapping projection (with process_input_state as sender, and mechanism as receiver)

        If len(Process.input) == len(mechanism.variable):
            - create one projection for each of the mechanism.inputState(s)
        If len(Process.input) == 1 but len(mechanism.variable) > 1:
            - create a projection for each of the mechanism.inputStates, and provide Process.input.value to each
        If len(Process.input) > 1 but len(mechanism.variable) == 1:
            - create one projection for each Process.input value and assign all to mechanism.inputState
        Otherwise,  if len(Process.input) != len(mechanism.variable) and both > 1:
            - raise exception:  ambiguous mapping from Process input values to mechanism's inputStates

        :param mechanism:
        :return:
        """

        # Convert Process input to 2D np.array
        process_input = convert_to_np_array(self.variable,2)

        # Get number of Process inputs
        num_process_inputs = len(process_input)

        # Get number of mechanism.inputStates
        #    - assume mechanism.variable is a 2D np.array, and that
        #    - there is one inputState for each item (1D array) in mechanism.variable
        num_mechanism_input_states = len(mechanism.variable)

        # There is a mismatch between number of Process inputs and number of mechanism.inputStates:
        if num_process_inputs > 1 and num_mechanism_input_states > 1 and num_process_inputs != num_mechanism_input_states:
            raise ProcessError("Mismatch between number of input values ({0}) for {1} and "
                               "number of inputStates ({2}) for {3}".format(num_process_inputs,
                                                                            self.name,
                                                                            num_mechanism_input_states,
                                                                            mechanism.name))

        # Create input state for each item of Process input, and assign to list
        for i in range(num_process_inputs):
            process_input_state = ProcessInputState(owner=self,
                                                    variable=process_input[i],
                                                    prefs=self.prefs)
            self.processInputStates.append(process_input_state)

        from PsyNeuLink.Functions.Projections.Mapping import Mapping

        # If there is the same number of Process input values and mechanism.inputStates, assign one to each
        if num_process_inputs == num_mechanism_input_states:
            for i in range(num_mechanism_input_states):
                # Insure that each Process input value is compatible with corresponding variable of mechanism.inputState
                if not iscompatible(process_input[i], mechanism.variable[i]):
                    raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                       "variable for corresponding inputState of {3}".
                                       format(i, process_input[i], self.name, mechanism.name))
                # Create Mapping projection from Process input state to corresponding mechanism.inputState
                Mapping(sender=self.processInputStates[i],
                        receiver=list(mechanism.inputStates.items())[i][1],
                        name=self.name+'_Input Projection',
                        context=context)
                if self.prefs.verbosePref:
                    print("Assigned input value {0} ({1}) of {2} to corresponding inputState of {3}".
                          format(i, process_input[i], self.name, mechanism.name))

        # If the number of Process inputs and mechanism.inputStates is unequal, but only a single of one or the other:
        # - if there is a single Process input value and multiple mechanism.inputStates,
        #     instantiate a single Process input state with projections to each of the mechanism.inputStates
        # - if there are multiple Process input values and a single mechanism.inputState,
        #     instantiate multiple Process input states each with a projection to the single mechanism.inputState
        else:
            for i in range(num_mechanism_input_states):
                for j in range(num_process_inputs):
                    if not iscompatible(process_input[j], mechanism.variable[i]):
                        raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                           "variable ({3}) for inputState {4} of {5}".
                                           format(j, process_input[j], self.name,
                                                  mechanism.variable[i], i, mechanism.name))
                    # Create Mapping projection from Process buffer_intput_state to corresponding mechanism.inputState
                    Mapping(sender=self.processInputStates[j],
                            receiver=list(mechanism.inputStates.items())[i][1],
                            name=self.name+'_Input Projection')
                    if self.prefs.verbosePref:
                        print("Assigned input value {0} ({1}) of {2} to inputState {3} of {4}".
                              format(j, process_input[j], self.name, i, mechanism.name))

        mechanism.receivesProcessInput = True

    def assign_input_values(self, input, context=NotImplemented):
        """Validate input, assign each item (1D array) in input to corresponding process_input_state

        Returns converted version of input

        Args:
            input:

        Returns:

        """
        # Validate input
        if input is NotImplemented:
            input = self.variableInstanceDefault
            if (self.prefs.verbosePref and
                    not (context is NotImplemented or kwFunctionInit in context)):
                print("- No input provided;  default will be used: {0}")

        else:
            # MODIFIED 8/19/16 OLD:
            # PROBLEM: IF INPUT IS ALREADY A 2D ARRAY OR A LIST OF ITEMS, COMPRESSES THEM INTO A SINGLE ITEM IN AXIS 0
            # input = convert_to_np_array(input, 2)
            # MODIFIED 8/19/16 NEW:
            # Insure that input is a list of 1D array items, one for each processInputState
            # If input is a single number, wrap in a list
            from numpy import ndarray
            if isinstance(input, numbers.Number) or (isinstance(input, ndarray) and input.ndim == 0):
                input = [input]
            # If input is a list of numbers, wrap in an outer list (for processing below)
            if all(isinstance(i, numbers.Number) for i in input):
                input = [input]

        if len(self.processInputStates) != len(input):
            raise ProcessError("Length ({}) of input to {} does not match the number "
                               "required for the inputs of its origin mechanisms ({}) ".
                               format(len(input), self.name, len(self.processInputStates)))

        # Assign items in input to value of each process_input_state
        for i in range (len(self.processInputStates)):
            self.processInputStates[i].value = input[i]

        return input

    def instantiate_deferred_inits(self, context=NotImplemented):
        """Instantiate any objects in the Process that have deferred their initialization

        Description:
            go through mechanismList in reverse order of configuration since
                learning signals are processed from the output (where the training signal is provided) backwards
            exhaustively check all of components of each mechanism,
                including all projections to its inputStates and parameterStates
            initialize all items that specified deferred initialization
            construct a monitoringMechanismList of mechanism tuples (mech, params, phase_spec):
                assign phase_spec for each MonitoringMechanism = self.phaseSpecMax + 1 (i.e., execute them last)
            add monitoringMechanismList to the Process' mechanismList
            assign input projection from Process to first mechanism in monitoringMechanismList

        IMPLEMENTATION NOTE: assume that the only projection to a projection is a LearningSignal

        IMPLEMENTATION NOTE: this is implemented to be fully general, but at present may be overkill
                             since the only objects that currently use deferred initialization are LearningSignals
        """

        # For each mechanism in the Process, in backwards order through its mechanismList
        for item in reversed(self.mechanismList):
            mech = item[OBJECT]
            mech.deferred_init()

            # For each inputState of the mechanism
            for input_state in mech.inputStates.values():
                input_state.deferred_init()
                self.instantiate_deferred_init_projections(input_state.receivesFromProjections, context=context)

            # For each parameterState of the mechanism
            for parameter_state in mech.parameterStates.values():
                parameter_state.deferred_init()
                self.instantiate_deferred_init_projections(parameter_state.receivesFromProjections)

        # If there are monitoringMechanisms
        if self.monitoringMechanismList:

            # They have been assigned self.phaseSpecMax+1, so increment self.phaseSpeMax
            self.phaseSpecMax = self.phaseSpecMax + 1

            # Create ProcessInputState for target of output MonitoringMechanism (first one in monitoringMechanismList)
            from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import kwComparatorTarget
            monitoring_mechanism_target = self.monitoringMechanismList[0][OBJECT].inputStates[kwComparatorTarget]

            process_input_state = ProcessInputState(owner=self,
                                                    variable=monitoring_mechanism_target.variable,
                                                    prefs=self.prefs,
                                                    name=kwComparatorTarget)
            self.processInputStates.append(process_input_state)

            # Extend Process variable to include target
            # ---------------
            # MODIFIED 8/19/16:
            # input = np.concatenate((self.variable, np.atleast_2d(monitoring_mechanism_target.variable)))

            input = list(self.variable)
            input.extend(np.atleast_2d(monitoring_mechanism_target.variable))
            input = np.array(np.array(input))
            self.assign_defaults(variable=input)

            # Add Mapping projection from the ProcessInputState to MonitoringMechanism's target inputState
            from PsyNeuLink.Functions.Projections.Mapping import Mapping
            Mapping(sender=process_input_state,
                    receiver=monitoring_mechanism_target,
                    name=self.name+'_Input Projection to '+monitoring_mechanism_target.name)

            # Add monitoringMechanismList to mechanismList
            self.mechanismList.extend(self.monitoringMechanismList)

    def instantiate_deferred_init_projections(self, projection_list, context=NotImplemented):

        # For each projection in the list
        for projection in projection_list:
            projection.deferred_init()

            # For each parameter_state of the projection
            try:
                for parameter_state in projection.parameterStates.values():

                    # Initialize each LearningSignal projection
                    for learning_signal in parameter_state.receivesFromProjections:
                        learning_signal.deferred_init(context=context)
            # Not all Projection subclasses instantiate parameterStates
            except AttributeError as e:
                pass

            # Check if projection has monitoringMechanism attribute
            try:
                monitoring_mechanism = projection.monitoringMechanism
            except AttributeError:
                pass
            else:
                # If a monitoringMechanism has been assigned, pack in tuple and assign to monitoringMechanismList
                if monitoring_mechanism:
                    mech_tuple = (monitoring_mechanism, None, self.phaseSpecMax+1)
                    self.monitoringMechanismList.append(mech_tuple)

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

        IMPLEMENTATION NOTE:
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

        report_output = self.prefs.reportOutputPref and not context is NotImplemented and kwExecuting in context


        # FIX: CONSOLIDATE/REARRANGE assign_input_values, check_args, AND ASIGNMENT OF input TO self.variable
        # FIX: (SO THAT assign_input_value DOESN'T HAVE TO RETURN input

        input = self.assign_input_values(input=input, context=context)

        self.check_args(input,runtime_params)

        if time_scale is NotImplemented:
            self.timeScale = TimeScale.TRIAL

        if (kwExecuting in context):  # Note: not necessarily so, as execute method is also called for validation
            if self.prefs.reportOutputPref:
                print("\n\n****************************************\n\n{0} executing with:\n- configuration: [{1}]".
                      # format(self.name, re.sub('[\[,\],\n]','',str(self.configurationMechanismNames))))
                      format(self.name, re.sub('[\[,\],\n]','',str(self.mechanismNames))))

        # Use value of Process as input to first Mechanism in Configuration
        self.variable = input

        # Report input if reporting preference is on and this is not an initialization run
        # if self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context):
        if report_output:
            print("- input: {1}".format(self.name, re.sub('[\[,\],\n]','',str(self.variable))))

        #region EXECUTE EACH MECHANISM
        # Execute each Mechanism in the configuration, in the order listed
        for i in range(len(self.mechanismList)):
            mechanism, params, phase_spec = self.mechanismList[i]

            # FIX:  DOES THIS BELONG HERE OR IN SYSTEM?
            # CentralClock.time_step = i

            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
            mechanism.execute(time_scale=self.timeScale,
                              runtime_params=params,
                              context=context)

            # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR THE MECHANISM ITSELF
            # if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
            if report_output:
                print("\n{0} executed {1}:\n- output: {2}\n\n--------------------------------------".
                      format(self.name,
                             mechanism.name,
                             re.sub('[\[,\],\n]','',
                                    str(mechanism.outputState.value))))
            if not i:
                # Zero input to first mechanism after first run (in case it is repeated in the configuration)
                # IMPLEMENTATION NOTE:  in future version, add option to allow Process to continue to provide input
                self.variable = self.variable * 0
            i += 1
        #endregion

        #region EXECUTE LearningSignals
        # Update each LearningSignal, beginning projection(s) to last Mechanism in mechanismList, and working backwards
        for item in reversed(self.mechanismList):
            mech = item[OBJECT]
            params = item[PARAMS]

            # For each inputState of the mechanism
            for input_state in mech.inputStates.values():
                # For each projection in the list
                for projection in input_state.receivesFromProjections:
                    # For each parameter_state of the projection
                    try:
                        for parameter_state in projection.parameterStates.values():
                            # Call parameter_state.update with kwLearning to update LearningSignals
                            # Note: do this rather just calling LearningSignals directly
                            #       since parameter_state.update handles parsing of LearningSignal-specific params
                            # MODIFIED 9/4/16:
                            # FIX: ADD kwLearning TO CONTEXT HERE (SO LEARNING INIT METHODS CAN SEE IT??)
                            context = context + kwSeparatorBar + kwLearning
                            parameter_state.update(params=params, time_scale=TimeScale.TRIAL, context=context)
                            # for learning_signal in parameter_state.receivesFromProjections:
                            #     learning_signal.update(params=params, time_scale=time_scale, context=context)

                    # Not all Projection subclasses instantiate parameterStates
                    except AttributeError as e:
                        pass
        #endregion

        # if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
        if report_output:
            print("\n{0} completed:\n- output: {1}\n\n*********************************************\n".
                  format(self.name,
                                                           re.sub('[\[,\],\n]','',str(self.outputState.value))))
        return self.outputState.value

    def get_configuration(self):
        """Return configuration (list of Projection tuples)
        The configuration is an ordered list of Project tuples, each of which contains:
             sender (State object)
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


class ProcessInputState(OutputState):
    """Represent input to process and provide to first Mechanism in Configuration

    Each instance encodes an item of the Process input (one of the 1D arrays in the 2D np.array input) and provides
        the input to a Mapping projection to one or more inputStates of the first Mechanism in the Configuration;
        see Process Description for mapping when there is more than one Process input value and/or Mechanism inputState

     Notes:
      * Declared as sublcass of OutputState so that it is recognized as a legitimate sender to a Projection
           in Projection.instantiate_sender()
      * self.value is used to represent input to Process provided as variable arg on command line

    """
    def __init__(self, owner=None, variable=NotImplemented, name=NotImplemented, prefs=NotImplemented):
        """Pass variable to mapping projection from Process to first Mechanism in Configuration

        :param variable:
        """
        if name is NotImplemented:
            self.name = owner.name + "_" + kwProcessInputState
        else:
            self.name = owner.name + "_" + name
        self.prefs = prefs
        self.sendsToProjections = []
        self.owner = owner
        self.value = variable


