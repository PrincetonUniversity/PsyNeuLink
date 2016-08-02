#
# *********************************************  Process ***************************************************************
#

import re
import math
import Functions
from Functions.ShellClasses import *
from Globals.Registry import register_category
from Functions.Mechanisms.Mechanism import Mechanism_Base
from Functions.Mechanisms.Mechanism import mechanism

# *****************************************    PROCESS CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# Labels for items in configuration entry tuples
OBJECT = 0
PARAMS = 1

ProcessRegistry = {}


class ProcessError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


# Process factory method:
def process(process_spec=NotImplemented, params=NotImplemented, context=NotImplemented):
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
        return Process_Base()

    # Can't be anything else, so return empty
    else:
        return None


kwProcessInputState = 'ProcessInputState'
from Functions.States.MechanismOutputState import MechanismOutputState

# DOCUMENT:  HOW DO MULTIPLE PROCESS INPUTS RELATE TO # OF INPUTSTATES IN FIRST MECHANISM
#            WHAT HAPPENS IF LENGTH OF INPUT TO PROCESS DOESN'T MATCH LENGTH OF VARIABLE FOR FIRST MECHANISM??


class Process_Base(Process):
# DOCUMENT:  CONFIGURTION FORMAT:  (Mechanism <, PhaseSpec>) <, Projection,> (Mechanism <, PhaseSpec>)
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
        A Process is defined by a kwConfiguration param (list of Mechanisms) and a time scale.  Executing a Process
         executes the Mechanisms in the order that they appear in the configuration.

    Instantiation:
        A Process can be instantiated in one of two ways:
            - by a direct call to Process()
            [TBI: - in a list to a call to System()]
        A Process instantiates its configuration by assigning:
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
# DOCUMENT:  UPDATE TO INCLUDE Mechanism, Projection, Mechanism FORMAT, AND (Mechanism, Cycle) TUPLE
            kwConfiguration (list): (default: single Mechanism_Base.defaultMechanism)
                Each config_entry must be one of the following, that is used to instantiate the mechanisms in the list:
                    + Mechanism object
                    + Mechanism type (class) (e.g., DDM)
                    + descriptor keyword for a Mechanism type (e.g., kwDDM)
                    + specification dict for Mechanism; the dict can have the following entries (see Mechanism):
                        + kwMechanismType (Mechanism subclass): if absent, Mechanism_Base.defaultMechanism is used
                        + entries with keys = standard args of Mechanism.__init__:
                            "input_template":<value>
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
                                + kwInputStateParams:<dict>
                                + kwMechanismParameterStateParams:<dict>
                           [TBI + kwMechanismOutputStateParams:<dict>]
                                - each dict will be passed to the corresponding State
                                - params can be any permissible executeParamSpecs for the corresponding State
                                - dicts can contain the following embedded dicts:
                                    + kwExecuteMethodParams:<dict>:
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
            • input specified as arg in execution of Process is provided as input to first mechanism in configuration
            • output of last mechanism in configuration is assigned as Process.ouputState.value
            • DefaultController is executed before execution of each mechanism in the configuration
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
# DOCUMENT:  EXPLAIN (Mechanism <, Cycle>) <, Projection,> (Mechanism <, Cycle>) FORMAT
        + configuration (list): set in params[kwConfiguration]
            an ordered list of tuples that defines how the process is carried out;
                (default: the default mechanism for the Mechanism class — currently: DDM)
                Note:  this is constructed from the kwConfiguration param, which may or may not contain tuples;
                       all entries of kwConfiguration param are converted to tuples for self.configuration
                       for entries that are not tuples, None is used for the param (2nd) item of the tuple
# DOCUMENT: THESE HAVE BEEN REPLACED BY processInputStates (BELOW)
        # + sendsToProjections (list)           | these are used to instantiate a projection from Process
        # + ownerMechanism (None)               | to first mechanism in the configuration list
        # + value (value)                       | value is used to specify input to Process;
        #                                       | it is zeroed after executing the first item in the configuration
        + processInputStates (MechanismOutputState:
            instantiates projection(s) from Process to first Mechanism in the configuration
        + outputState (MechanismsState object) - reference to MechanismOutputState of last mechanism in configuration
            updated with output of process each time process.execute is called
        + phaseSpecMax (int) - integer component of maximum phaseSpec for Mechanisms in configuration
        + system (System) - System to which Process belongs
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
        self.processInputStates = []
        self.phaseSpecMax = 0
        
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
                  format(self.name, self.mechanism_names.__str__().strip("[]")))

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
        # If validation pref is set, instantiate and execute the Process
        if self.prefs.paramValidationPref:
            super(Process_Base, self).instantiate_execute_method(context=context)
        # Otherwise, just set Process output info to the corresponding info for the last mechanism in the configuration
        else:
            self.value = self.configuration[-1][OBJECT].outputState.value

# DOCUMENTATION:
#         Uses paramClassDefaults[kwConfiguration] == [Mechanism_Base.defaultMechanism] as default
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
                    use kwFullConnectivityMatrix (full connectivity matrix with unit weights) if the lengths are not equal

        :param context:
        :return:
        """
# IMPLEMENT:  NEW -------------------------------------------------------------------
        # DOCUMENT:
        # Each item in Configuration can be a Mechanism or Projection object, class ref, or specification dict,
        #     str as name for a default Mechanism,
        #     keyword (kwIdentityMatrix or kwFullConnectivityMatrix) as specification for a default Projection,
        #     or a tuple with any of the above as the first item and a param dict as the second

        configuration = self.paramsCurrent[kwConfiguration]
        self.mechanism_list = []
        self.mechanism_names = []

# FIX: LENGTHEN TUPLE INSTANTIATION HERE (LEN = 3) TO ACCOMODATE phaseSpec, AND ADD TO PARSE BELOW;
# FIX:  DEFAULT: 1 (UPDATE FULLY EVERY CYCLE)
# IMPLEMENTATION NOTE:  for projections, 2nd and 3rd items of tuple are ignored

        #region STANDARDIZE ENTRY FORMAT

        # Convert all entries to (item, params, phaseSpec) tuples, padded with None for absent params and/or phaseSpec
        for i in range(len(configuration)):
            config_item = configuration[i]
            if isinstance(config_item, tuple):
                # If the tuple has only one item, assume it is a Mechanism or Projection specification; pad with None
                if len(config_item) is 1:
                    configuration[i] = (config_item[0], None, None)
                # If the tuple has two items, check whether second item is a params dict or a phaseSpec
                #    and assign it to the appropriate position in the tuple, padding other with None
                if len(config_item) is 2:
                    if isinstance(config_item[1], dict):
                        configuration[i] = (config_item[0], config_item[1], None)
                    elif isinstance(config_item[1], (int, float)):
                        configuration[i] = (config_item[0], None, config_item[1])
                    else:
                        raise ProcessError("Second item of tuple ((0}) in item {1} of configuration for {2}"
                                           " is neither a params dict nor phaseSpec (int or float)".
                                           format(config_item[1], i, self.name))
                if len(config_item) > 3:
                    raise ProcessError("The tuple for item {0} of configuration for {1} has more than three items {2}".
                                       format(i, self.name, config_item))
            else:
                # Convert item to tuple, padded with None
                configuration[i] = (configuration[i], None, None)
        #endregion

        #region VALIDATE CONFIGURATION AND PARSE AND INSTANTIATE MECHANISM ENTRIES

        # - make sure first entry is not a Projection
        # - make sure Projection entries do NOT occur back-to-back (i.e., no two in a row)
        # - instantiate Mechanism entries

        previous_item_was_projection = False
        from Functions.Projections.Projection import Projection_Base

        for i in range(len(configuration)):
            # MODIFIED 7/1/16
            # item, params = configuration[i]
            item, params, phase_spec = configuration[i]

            # Get max phaseSpec for Mechanisms in configuration
            if not phase_spec:
                phase_spec = 0
            self.phaseSpecMax = int(max(math.floor(float(phase_spec)), self.phaseSpecMax))

            #region VALIDATE PLACEMENT OF PROJECTION ENTRIES
            # Can't be first entry, and can never have two in a row

            # Config entry is a Projection
            if Projection_Base.is_projection_spec(item):
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

            #region INSTANTIATE MECHANISM
            # Must do this before assigning projections (below)
            # Mechanism entry must be a Mechanism object, class, specification dict, str, or (Mechanism, params) tuple
            # Don't use params item of tuple (if present) to instantiate Mechanism, as they are runtime only params

            # Entry is NOT already a Mechanism object
            if not isinstance(mech, Mechanism):
                # Note: need full pathname for mechanism factory method, as "mechanism" is used as local variable below
                mech = Functions.Mechanisms.Mechanism.mechanism(mech, context=context)
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
            # Add entry to mechanism_list and name to mechanism_names list
            mech.phaseSpec = phase_spec
            self.mechanism_list.append(configuration[i])
            self.mechanism_names.append(mech.name)
            #endregion
        #endregion

        # Assign process outputState to last mechanisms in configuration
        self.firstMechanism = configuration[0][OBJECT]
        self.lastMechanism = configuration[-1][OBJECT]
        self.outputState = self.lastMechanism.outputState

        #region PARSE, INSTANTIATE AND ASSIGN PROJECTION ENTRIES
        from Functions.Projections.Mapping import Mapping
        for i in range(len(configuration)):
            item, params, phase_spec = configuration[i]

            #region FIRST ENTRY

            # Must be a Mechanism (enforced above)
            # Assign input(s) from Process to it if it doesn't already have any
            if i == 0:
                # Relabel for clarity
                mechanism = item

                # Check if first Mechanism already has any projections
                if item.inputState.receivesFromProjections:
                    # Check where the projection(s) is/are from and, if verbose pref is set, issue appropriate warnings
                    for projection in mechanism.inputState.receivesFromProjections:

                        # Projection to first Mechanism in Configuration comes from a Process input
                        if isinstance(projection.sender, ProcessInputState):
                            # If it is:
                            # (A) from self, ignore
                            # (B) from another Process, warn if verbose pref is set
                            if not projection.sender.ownerMechanism is self:
                                if self.prefs.verbosePref:
                                    print("WARNING: {0} in configuration for {1} already has an input from {2} "
                                          "that will be used".
                                          format(mechanism.name, self.name, projection.sender.ownerMechanism.name))
                            #     self.assign_process_input_projections(mechanism)
                            # continue

                        # (C) Projection to first Mechanism in Configuration comes from one in the Process' mechanism_list;
                        #     so warn if verbose pref is set
                        if projection.sender.ownerMechanism in self.mechanism_list:
                            if self.prefs.verbosePref:
                                print("WARNING: first mechanism ({0}) in configuration for {1} receives "
                                      "a (recurrent) projection from another mechanism {2} in {1}".
                                      format(mechanism.name, self.name, projection.sender.ownerMechanism.name))
                                # self.assign_process_input_projections(mechanism)
                                # continue

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
                                          format(mechanism.name, self.name, projection.sender.ownerMechanism.name))
                                # self.assign_process_input_projections(mechanism)
                                # continue
                            else:
                                # Process IS being implemented as part of a System,
                                if system:
                                    # (E) Projection is from another Mechanism in the System
                                    #    (most likely the last in a previous Process)
                                    if mechanism in system.mechanisms:
                                        # self.assign_process_input_projections(mechanism)
                                        # continue
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
                                                     projection.sender.ownerMechanism.name,
                                                     context.name))
                                    #     self.assign_process_input_projections(mechanism)
                                    # continue
                                # Process is being implemented in something other than a System
                                #    so warn (irrespecive of verbose)
                                else:
                                    print("WARNING:  Process ({0}) being instantiated in context "
                                                       "({1}) other than a System ".format(self.name, context))

                # First entry does not have any projections, so assign relevant one(s)
                else:
                    pass

                # Assign input projection from Process
                self.assign_process_input_projections(mechanism)
                continue
            #endregion

            #region SUBSEQUENT ENTRIES

            # Item is a Mechanism
            if isinstance(item, Mechanism):

                preceding_item = configuration[i-1][OBJECT]

                # If preceding entry was a projection no need to do anything
                #    (as the current Mechanism should have already been assigned as the receiver)
                if isinstance(preceding_item, Projection):
                    continue

                # Preceding item was a Mechanism, so check if a Projection needs to be instantiated between them
                # Check if Mechanism already has a projection from the preceding Mechanism, by confirming that the
                #    preceding mechanism is not the sender of any projections received by the current one's inputState
# FIX: THIS SHOULD BE DONE FOR ALL INPUTSTATES
# FIX: POTENTIAL PROBLEM - EVC *CAN* HAVE MULTIPLE PROJECTIONS FROM (DIFFERENT outputStates OF) THE SAME MECHANISM
                if not (any(preceding_item == projection.sender.ownerMechanism
                            for projection in item.inputState.receivesFromProjections)):
                    # It is not, so instantiate mapping projection from preceding mechanism to current one:
                    # Note:
                    #   if len(preceding_item.value) == len(item.variable), the identity matrix will be used
                    #   if the lengths are not equal, the unit full connectivity matrix will be used
                    #   (see LinearMatrix Utility Function for details)
                    Mapping(sender=preceding_item, receiver=item)
                    if self.prefs.verbosePref:
                        print("Mapping projection added from mechanism {0} to mechanism {1} in configuration of {2}".
                              format(preceding_item.name, mech.name, self.name))

            # Item should be a Projection
            else:
                # Instantiate Projection, assigning mechanism in previous entry as sender and next one as receiver
                # IMPLEMENTATION NOTE:  FOR NOW:
                #    - ASSUME THAT PROJECTION SPECIFICATION (IN item) IS ONE OF THE FOLLOWING:
                #        + Projection object
                #        + Matrix object
                #        +  Matrix keyword (kwIdentityMatrix or kwFullConnectivityMatrix)
                #    - params IS IGNORED
                # FIX: PARSE/VALIDATE PROJECTION SPEC (ITEM PART OF TUPLE) HERE: CLASS, OBJECT, DICT, STR, TUPLE??
                # IMPLEMENT: MOVE State.instantiate_projections(), check_projection_receiver()
                #            and parse_projection_ref() all to Projection_Base.__init__() and call that
                #           VALIDATION OF PROJECTION OBJECT:
                #                MAKE SURE IT IS A Mapping PROJECTION
                #                CHECK THAT SENDER IS configuration[i-1][OBJECT]
                #                CHECK THAT RECEVIER IS configuration[i+1][OBJECT]
                if (isinstance(item, np.matrix) or
                        (isinstance(item, np.ndarray) and item.ndim == 2) or
                            kwIdentityMatrix in item or
                            kwFullConnectivityMatrix in item):
                    projection_params = {kwExecuteMethodParams: {kwMatrix: item}}
                    projection = Mapping(sender=configuration[i-1][OBJECT],
                                         receiver=configuration[i+1][OBJECT],
                                         params=projection_params)
                    # Reassign Configuration entry
                    #    with Projection as OBJECT item and original params as PARAMS item of the tuple
                    # IMPLEMENTATION NOTE:  params is currently ignored
                    configuration[i] = (projection, params)
                else:
                    raise ProcessError("Item {0} ({1}) of configuration for {2} is not "
                                       "a valid mechanism or projection specification".format(i, item, self.name))
            #endregion

        #endregion

        self.configuration = configuration

    def assign_process_input_projections(self, mechanism):
        """Create projection(s) for each item in input item to inputState(s) of the specified Mechanism

        For each item in input:
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

        from Functions.Projections.Mapping import Mapping

        # If there is the same number of Process input values and mechanism.inputStates, assign one to each
        if num_process_inputs == num_mechanism_input_states:
            for i in range(num_mechanism_input_states):
                # Insure that each Process input value is compatible with corresponding variable of mechanism.inputState
                if not iscompatible(process_input[i], mechanism.variable[i]):
                    raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                       "variable for corresponding inputState of {3}".
                                       format(i, process_input[i], self.name, mechanism.name))
                # Create Mapping projection from Process input state to corresponding mechanism.inputState
                Mapping(sender=self.processInputStates[i], receiver=list(mechanism.inputStates.items())[i][1])
# DOCUMENTATION:  INTEL HACKATHON EXAMPLE
                # Mapping(sender=self.processInputStates[i], receiver=mechanism.inputStates.items()[i][1])
                if self.prefs.verbosePref:
                    print("Assigned input value {0} ({1}) of {2} to corresponding inputState of {3}".
                          format(i, process_input[i], self.name, mechanism.name))

        # If the number of Process inputs and mechanism.inputStates is unequal, but only a single of one or the other
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
                    Mapping(sender=self.processInputStates[j], receiver=list(mechanism.inputStates.items())[i][1])
                    if self.prefs.verbosePref:
                        print("Assigned input value {0} ({1}) of {2} to inputState {3} of {4}".
                              format(j, process_input[j], self.name, i, mechanism.name))

        mechanism.receivesProcessInput = True



    def assign_input_values(self, input, context=NotImplemented):
        """Validate input, assign each item (1D np.array) in input to corresponding process_input_state

        Returns converted version of inpu

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
            input = convert_to_np_array(input, 2)

        # MODIFIED 7/26/16 NEW:
        if len(self.processInputStates) != len(input):
            raise ProcessError("Length ({}) of input to {} does not match the number "
                               "required for the inputs of its origin mechanisms ({}) ".
                               format(len(input), self.name, len(self.processInputStates)))
        # MODIFIED 7/26/16 END

        # Assign items in input to value of each process_input_state
        for i in range (len(self.processInputStates)):
            self.processInputStates[i].value = input[i]
        # Validate input
        if input is NotImplemented:
            input = self.variableInstanceDefault
            if (self.prefs.verbosePref and
                    not (context is NotImplemented or kwFunctionInit in context)):
                print("- No input provided;  default will be used: {0}")
        else:
            input = convert_to_np_array(input, 2)


        # Assign items in input to value of each process_input_state
        for i in range (len(self.processInputStates)):
            self.processInputStates[i].value = input[i]

        return input


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


        # FIX: CONSOLIDATE/REARRANGE assign_input_values, check_args, AND ASIGNMENT OF input TO self.variabe
        # FIX: (SO THAT assign_input_value DOESN'T HAVE TO RETURN input

        input = self.assign_input_values(input=input, context=context)

        self.check_args(input,runtime_params)

        if time_scale is NotImplemented:
            self.timeScale = TimeScale.TRIAL

        if (kwExecuting in context):  # Note: not necessarily so, as execute method is also called for validation
            if self.prefs.reportOutputPref:
                print("\n{0} executing with:\n- configuration: [{1}]".
                      # format(self.name, re.sub('[\[,\],\n]','',str(self.configurationMechanismNames))))
                      format(self.name, re.sub('[\[,\],\n]','',str(self.mechanism_names))))

        # Use value of Process as input to first Mechanism in Configuration
        self.variable = input

        # Report input if reporting preference is on and this is not an initialization run
        # if self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context):
        if report_output:
            print("- input: {1}".format(self.name, re.sub('[\[,\],\n]','',str(self.variable))))

        #region EXECUTE EACH MECHANISM
        # Execute each Mechanism in the configuration, in the order listed
        for i in range(len(self.mechanism_list)):
            # MODIFIED 7/1/16
            # mechanism, params = self.mechanism_list[i]
            mechanism, params, phase_spec = self.mechanism_list[i]
        # i = 0 # Need to use this, as can't use index on mechanism since it may be repeated in the configuration
        # for mechanism, params in self.configuration:

            # FIX:  DOES THIS BELONG HERE OR IN SYSTEM?
            # CentralClock.time_step = i

            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
            mechanism.update(time_scale=self.timeScale,
                             runtime_params=params,
                             context=context)
            # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR THE MECHANISM ITSELF
            # if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
            if report_output:
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
        if report_output:
            print("\n{0} completed:\n- output: {1}".format(self.name,
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


