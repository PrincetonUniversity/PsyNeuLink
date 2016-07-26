#
# **************************************************  ToDo *************************************************************
#
#region PY QUESTIONS: --------------------------------------------------------------------------------------------------
# QUESTION:  how to initialize a numpy array with a null value, and then assign in for loop: np.empty
#endregion
# -------------------------------------------------------------------------------------------------

#region DEPENDENCIES: -----------------------------------------------------------------------------------------------
#
# toposort.py
# wfpt.py
# mpi4py.py
#
#region BRYN: -------------------------------------------------------------------------------------------------------
#
# - ABC
# - params dict vs. args vs. **kwargs:  FIX: LOOK AT BRYN'S CHANGES TO isCompatible
# - FIX: LOOK AT HIS IMPLEMENTATION OF SETTER FOR @ClassProperty
# - QUESTION: CAN ERRORS IN TypeVar CHECKING BE CAPTURED AND CUSTOMIZED?
#            (TO PROVIDE MORE INFO THAN JUST THE ERROR AND WHERE IT OCCURRED (E.G., OTHER OBJECTS INVOLVED)
# - Revert all files to prior commit in PyCharm (VCS/Git/Revert command?)
# - kwNotation:  good for code but bad for script
#
#endregion

#region EVC MEETING: -------------------------------------------------------------------------------------------------------
#
#
# CLEANUP: underscore format used for local variables and methods (e.g., activationVector -> activation_vector)
#          camelback format used for attributes (lower case initial letter)
#                                    and keywords and class names (capitalized initial letter)
# FIX: HOW IS THIS DIFFERENT THAN LENGTH OF self.variable
#         + kwTransfer_NUnits (float): (default: Transfer_DEFAULT_NUNITS
#             specifies number of units (length of input array)
# IMPLEMENT: when instantiating a ControlSignal:
#                   include kwDefaultController as param for assigning sender to DefaultController
#                   if it is not otherwise specified
# IMPLEMENT: Transfer Mechanism:  executeMethod determines form of transfer (linear, logistic, etc.)
#            param names:  gain/bias vs. slope/intercept vs. steepness/offset?
# QUESTION:  Revisit issue of update vs. execute for Mechanisms (e.g., Transfer Mechanism)
# IMPLEMENT: Consider renaming "Utility" to "UtilityFunction":
#                   UtilityFunction seems a bit redundant (since Utility is a subclass of Function),
#                   but it is more descriptive
# IMPLEMENT: Learning objects:  Comparator Mechanism and Training Projection (see LEARNING below)
#              - what should the default inputState for kwResponseSignal be?
#              - what should the default inputState for kwTrainingSignal be?

#endregion

#region CURRENT: -------------------------------------------------------------------------------------------------------
#
# 7/25/16:
#
# FIX handling of inputStates (kwComparatorSample and kwComparatorTarget) in LinearComparator:
#              requirecParamClassDefaults
#              instantiate_attributes_before_execute_method
# FIX: DISABLE MechanismsParameterState execute Method ASSIGNMENT IF PARAM IS AN OPERATION;  JUST RETURN THE OP
#
# 7/24/16:
#
# IMPLEMENT/CONFIRM HANDLNG OF outputs AND outputState(s).value:
#                     implement self.outputValue
#                     update it everytime outputState.value or any outputStates[].value is assigned
#                     simplify outputStateValueMapping by implementing a method
#                         that takes list of ouput indices and self.outputStates
#                     Replace  output = [None] * len(self.paramsCurrent[kwMechanismOutputStates])
#                        with  output = [None] * len(outputStates)

#                     implement in DDM, Transfer, and LinearComparator mechanisms (or in Mechanisms)

#
# FIX: IN COMPARATOR instantiate_attributes_before_execute_method:  USE ASSIGN_DEFAULT
# FIX: IMPLEMENT Types for paramClassDefaults AND USE FOR Comparator Mechanism
# FIX:  TEST FOR FUNCTION CATEGORY == TRANSFER
# TEST: RUN TIMING TESTS FOR paramValidationPref TURNED OFF

# IMPLEMENT: Comparator Processing Mechanism TYPE, LinearComparator SUBTYPE
# IMPLEMENT: Training Projection
# IMPLEMENT: Add Integrator as Type of Utility and move Integrator from Transfer to Integrator
# FIX:
        # FIX: USE LIST:
        #     output = [None] * len(self.paramsCurrent[kwMechanismOutputStates])
        # FIX: USE NP ARRAY
        #     output = np.array([[None]]*len(self.paramsCurrent[kwMechanismOutputStates]))

# IMPLEMENT: Consider renaming "Utility" to "UtilityFunction"

# 7/23/16:
#
# IMPLEMENT:  ProcessingMechanism class:
#                 move any properties/methods of mechanisms not used by SystemControlMechanisms to this class
#                 for methods: any that are overridden by SystemControlMechanism and that don't call super
#
# 7/20/16:
#
# IMPLEMENT: PreferenceLevel SUBTYPE
#             IMPLEMENT TYPE REGISTRIES (IN ADDITION TO CATEGORY REGISTRIES)
#             IMPLEMENT Utility Functions:  ADD PreferenceLevel.SUBTYPE with comments re: defaults, etc.
#
# IMPLEMENT: Process factory method:
#                 add name arg (name=)
#                 test params (in particular, kwConfig)
#                 test dict specification
# IMPLEMENT: Quote names of objects in report output
#
# 7/16/16:
#
# IMPLEMENT: make paramsCurrent a @property, and force validation on assignment if validationPrefs is set

# 7/15/16:
#
# IMPLEMENT: RE-INSTATE MechanismsList SUBCLASS FOR MECHANISMS (TO BE ABLE TO GET NAMES)
#             OR CONSTRUCT LIST FOR system.mechanisms.names
#
# 7/14/16:
#
# FIX: IF paramClassDefault = None, IGNORE IN TYPING IN Function
# FIX: MAKE kwMonitoredOutputStates A REQUIRED PARAM FOR System CLASS
#      ALLOW IT TO BE:  MonitoredOutputStatesOption, Mechanism, MechanismOutputState or list containing any of those
# FIX: NEED TO SOMEHOW CALL validate_monitored_state FOR kwMonitoredOutputStates IN SYSTEM.params[]
# FIX: CALL instantiate_monitored_output_states AFTER instantiate_prediction_mechanism (SO LATTER CAN BE MONITORED)
# FIX: QUESTION:  WHICH SHOULD HAVE PRECEDENCE FOR kwMonitoredOutputStates default:  System, Mechanism or ConrolMechanism?
#
# 7/13/16:
#
# FIX/DOCUMENT:  WHY kwSystem: None FOR EVCMechanism AND DefaultControlMechanism [TRY REMOVING FROM BOTH]
# SEARCH & REPLACE: kwMechanismOutputStates -> kwOutputStates (AND SAME FOR inputStates)
# FIX: NAMING OF Input-1 vs. Reward (WHY IS ONE SUFFIXED AND OTHER IS NOT?):  Way in which they are registered?
#
# 7/10/16:
#
# IMPLEMENT: system.mechanismsList as MechanismList (so that names can be accessed)

# 7/9/16
#
# FIX: ERROR in "Sigmoid" script:
# Functions.Projections.Projection.ProjectionError: 'Length (1) of outputState for Process-1_ProcessInputState must equal length (2) of variable for Mapping projection'
#       PROBLEM: Mapping.instantiate_execute_method() compares length of sender.value, which for DDM is 3 outputStates
#                                                     with length of receiver, which for DDM is just a single inputState
#
#
# 7/4/16:
#
# IMPLEMENT: See *** items in System
# Fix: *** Why is self.execute not called in Mechanism.update??
#
# Fix Finish fixing LinearCombination:
#      (checking length of 1D constituents of 2D variable);
#      confirm that for 2D, it combines
#      consider doing it the other way, and called by projections
# Fix: ??Enforce 2D for parameters values:
# Fix:  DOCUMENT:
#       - If its a 1D vector, then just scale and offset, but don't reduce?
#       - So, the effect of reduce would only occur for 2D array of single element arrays
# Fix sigmoid range param problem (as above:  by enforcing 2D)
#
#
# CLEANUP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####

# --------------------------------------------
# FIX: FINISH UP executeOutoutMethodDefault -> self.value:
# 4) # # MODIFIED 6/14/16: QQQ - WHY DOESN'T THIS WORK HERE??
# --------------------------------------------

# CONVERSION TO NUMPY:
# FIX: CONSTRAINT CHECKING SHOULD BE DONE AS BEFORE, BUT ONLY ON .value 1D array
# --------------
# FIX (FUNCTIONS / LinearMatrix): IS THIS STILL NEEDED??  SHOULDN'T IT BE 1D ARRAY??
# FIX (FUNCTIONS / LinearMatrix): MOVE KEYWORD DEFINITIONS OUT OF CLASS (CONFUSING) AND REPLACE self.kwXXX with kwXXX
# -------------
# FIX (PROJECTION): FIX MESS CONCERNING VARIABLE AND SENDER ASSIGNMENTS:
#         SHOULDN'T variable_default HERE BE sender.value ??  AT LEAST FOR Mapping?, WHAT ABOUT ControlSignal??
#                     MODIFIED 6/12/16:  ADDED ASSIGNMENT ABOVE
#                                       (TO HANDLE INSTANTIATION OF DEFAULT ControlSignal SENDER -- BUT WHY ISN'T VALUE ESTABLISHED YET?
# --------------
# FIX: (MechanismOutputState 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to ownerMechanism.outputStates here (and removing from ControlSignal.instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlSignal.instantiate_sender),
# -------------
# FIX: CHECK FOR dtype == object (I.E., MIXED LENGTH ARRAYS) FOR BOTH VARIABLE AND VALUE REPRESENTATIONS OF MECHANISM)
# FIX: (CATEGORY CLASES): IMPLEMENT .metaValue (LIST OF 1D ARRAYS), AND USE FOR ControlSignal AND DDM EXTRA OUTPUTS
# FIX: IMPLEMENT HIERARCHICAL SETTERS AND GETTERS FOR .value AND .metavalues;  USE THEM TO:
#                                                                         - REPRESENT OUTPUT FORMAT OF executeMethod
#                                                                         - ENFORCE 1D DIMENSIONALITY OF ELEMENTS
#                                                                         - LOG .value AND .metavalues

# END CLEANUP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# TEST: [0, 0, 0] SHOULD GENERATE ONE LIST InputState BUT [[0], [0], [0]] SHOULD GENERATE THREE
# TEST: TRY Process with variableClassDefault = 0 and [0]

# FIX: 6.10.16
#     X Main.convert_to_np_array
#     * self.variable assignments in Mechanism (2D), MechanismStates and Projection (1D)
#     * Mechanism needs to override validate_variable to parse and assign multi-value variable to 2D ARRAY:
#         COORDINATE MULTI-VALUE VARIABLE (ONE FOR EACH INPUT STATE) WITH variable SPECIFIED IN kwInputState PARAM:
#         COMPARE LENGTH OF MECHANISM'S VARIABLE (I.E., #OF ARRAYS IN LIST) WITH kwInputstate:
#                        LENGTH OF EITHER LIST OF NAMES OR SPECIFICATION DICT (I.E., # ENTRIES)
#                        DO THIS IN INSTANTIATE_MECHANISM_STATE IF PARAM_STATE_IDENTIFIER IS MechanismInputState
#                        OR HAVE MechanismInputState OVERRIDE THE METHOD
#     * in Mechanism, somehow, convert output of execute method to 2D array (akin to variable) one for each outputstate
#     * constraint_values in Mechanism.instantiate_mechanismState_lists (2D)
#     * entries (for logs) in MechanismState.value setter (1D) and ControlSignal.update (1D)
#     * Add Function.metaValue as alternative to multple outputState:
#               - should parallel .value in overridable setter/getter structure
#               - should log specified values
#     * INSTANTIATE MECHANISM STATE SHOULD PARSE self.variable (WHICH IS FORCED TO BE A 2D ARRAY) INTO SINGLE 1D ARRAYS

# ************************************************************************************************
#
#
# FIX: SETTLE ON execute VS. update
# put function in .function or .update_function attrib, and implement self.execute
#  that calls .function or .update_function with inputState as variable and puts value in outputState
#
# IMPLEMENT: execute method FOR DefaultControlMechanism (EVC!!)
# TEST: DefaultController's ability to change DDM params
# IMPLEMENT: Learning projection (projection that has another projection as receiver)
#
# FROM EVC MEETING 5/31/16:
# FIX: CLARIFY SPECIFICATION OF PARAM DICTS
# FIX:  Syntax for assigning kwExeuteMethodParam for ParameterState at time of mechanism instantiation
#               PROBLEM 1:  for paramClassDefaults, must specify ParamValueProjection tuple,
#                           but not in Configuration, don't need to do that (can just use tuple notation)
#               SOLUTION:   get rid of ParamValueProjection tuple?
#
#               PROBLEM 2:  params (e.g., DriftRate) are specified as:
#                               kwExecuteMethodParams in paramClassDefaults and Mechanism declartion
#                               kwMechanismParameterStateParams in Process Configuration list
# CONFIRM:  Syntax to specify ModulationOperation for ParameterState at time of mechanism instantiation
# FIX: ConrolSignal.set_intensity SHOULD CHANGE paramInstanceDefaults
# CONFIRM:  ControlSignal.intensity GETS COMBINED WITH allocadtion_source USING ModulationOperation
# QUESTION: WHAT DOES SETTING ControlSignal.set_intensity() DO NOW??
# IMPLEMENT: ADD PARAM TO DDM (AKIN TO kwDDM_AnayticSolution) THAT SPECIFIES PRIMARY INPUTSTATE (i.e., DRIFT_RATE, BIAS, THRSHOLD)
#endregion

#region GENERAL: -------------------------------------------------------------------------------------------------------------
#
# - Register name:
#    PsyNeuLink
#    [PsyPy? PsyPyScope?  PyPsyScope?  PsyScopePy? NeuroPsyPy?  NeuroPsySpy]
#
# Search & Replace:
#   kwMechanismParameterState -> kwMechanismParameterStates
#   MechanismParamValueparamModulationOperation -> MechanismParamValueParamModulationOperation
#   ExecuteMethodParams -> MechanismParameterStates
#   InputStateParams, OutputStateParams and ParameterStateParams => <*>Specs
#   KwDDM_StartingPoint -> DDM_StartingPoint
#   CHANGE ALL VARIABLES FROM THEIR LOCAL NAMES (E.G., Allocation_Source, Input_value, etc) to variable
#   Projections: sendsTo and sendsFrom
#   "or isinstance(" -> use tuple
#   Change "baseValue" -> "instanceValue" for prefs
#   Change Utility Functoin "LinearCombination" -> "LinearCombination"
#   super(<class name>, self) -> super()
#
#    Terminology:
#        QUESTION:
#        execute method -> update function [match case] (since it can be standalone (e.g., provided as param)
#        update_* -> execute_*
#
# - FIX: GET RID OFF '-1' SUFFIX FOR CUSTOM NAMES (ONLY ADD SUFFIX FOR TWO OR MORE OF SAME NAME, OR FOR DEFAULT NAMES)
# - FIX: MAKE ORDER CONSISTENT OF params AND time_scale ARGS OF update() and execute()
#
# - IMPLEMENT: integrate logging and verbose using BrainIAK model:
#              no printing allowed in extensions
#              verbose statements are logged
#              log goes to screen by default
#              can define file to which log will go
#
# - IMPLEMENT: master registry of all Function objects
#
# - IMPLEMENT switch in __init__.py to suppress processing for scratch pad, etc.
#
# - IMPLEMENT Testing:
#     use instantiation sequence (in Main) to create test for each step
#
# - Fully implement logging
#    For both of the above:
#       use @property to determine whether current value should be set to local value, type, category or class default
# - Implement timing
# - implement **args (per MechanismState init)
# - MAKE SURE check_args IS CALLED IN execute
#
# - iscompatible:
# -   # MAKE SURE / i IN iscompatible THAT IF THE REFERENCE HAS ONLY NUMBERS, THEN numbers_only SHOULD BE SET
# -   Deal with int vs. float business in iscompatible (and Utility_Base functionOutputTypeConversion)
# -   Fix: Allow it to allow numbers and strings (as well as lists) by default
#     and then relax constraint to be numeric for MechanismInputState, MechanismOutputState and MechanismParameterState
#     in Mechanism.validate_params
# -   Implement: #  IMPLEMENTATION NOTE:  modified to allow numeric type mismatches; should be added as option in future
#
# IMPLEMENT: add params as args in calls to __init__() for Function objects (as alternative to using params[])
#
# MAKE CONSISTENT:  variable, value, and input
#
#
# - Registry:
#   why is LinearCombination Utility Functions registering an instanceCount of 12 but only 2 entries?
#   why is DDM registering as subclass w/o any instances?
#   why are SLOPE and INTERCEPT in same registry as MechanismStatess and Parameters?
#   IMPLEMENT: Registry class, and make <*>Registry dicts instances of it, and include prefs attribute
#
# IMPLEMENT: change context to Context namedtuple (declared in Globals.Keywords or Main):  (str, object)
#
#endregion

# region DEPENDENCIES:
#   - toposort
#   - mpi4py
#   - wfpt.py
# endregion

# region OPTIMIZATION:
#   - get rid of tests for PROGRAM ERROR
# endregion

#region DOCUMENT: ------------------------------------------------------------------------------------------------------------
#
#  CLEAN UP THE FOLLOWING
# - Combine "Parameters" section with "Initialization arguments" section in:
#              Utility, Mapping, ControlSignal, and DDM documentation:

# DOCUMENT: requiredParamClassDefaultTypes:  used for paramClassDefaults for which there is no default value to assign
# DOCUMENT: CHANGE MADE TO FUNCTION SUCH THAT paramClassDefault[param:NotImplemented] -> NO TYPE CHECKING
# DOCUMENT: EVC'S AUTOMATICALLY INSTANTIATED predictionMechanisms USURP terminalMechanism STATUS
#           FROM THEIR ASSOCIATED INPUT MECHANISMS (E.G., Reward Mechanism)
# DOCUMENT:  kwPredictionMechanismType IS A TYPE SPECIFICATION BECAUSE INSTANCES ARE
#                 AUTOMTICALLY INSTANTIATED BY EVMechanism AND THERE MAY BE MORE THAN ONE
# DOCUMENT:  kwPredictionMechanismParams, AND THUS kwMonitoredOutputStates APPLIES TO ALL predictionMechanisms
# DOCUMENT: System.mechanisms:  DICT:
#                KEY FOR EACH ENTRY IS A MECHANIMS IN THE SYSTEM
#                VALUE IS A LIST OF THE PROCESSES TO WHICH THE MECHANISM BELONGS
# DOCUMENT: MEANING OF / DIFFERENCES BETWEEN self.variable, self.inputValue, self.value and self.outputValue
# DOCUMENT: DIFFERENCES BETWEEN EVCMechanism.inputStates (that receive projections from monitored States) and
#                               EVCMechanism.MonitoredOutputStates (the terminal states themselves)
# DOCUMENT: CURRENTLY, PREDICTION MECHANISMS USE OUTPUT OF CORRESPONDING ORIGIN MECHANISM
#           (RATHER THAN THEIR INPUT, WHICH == INPUT TO THE PROCESS)
#           LATTER IS SIMPLEST AND PERHAPS CLOSER TO WHAT IS MOST GENERALLY THE CASE
#               (I.E., PREDICT STIMULUS, NOT TRANSFORMED VERSION OF IT)
#           CURRENT IMPLEMENTATION IS MORE GENERAL AND FLEXIBLE,
#                BUT REQUIRES THAT LinearMechanism (OR THE EQUIVALENT) BE USED IF PREDICTION SHOULD BE OF INPUT

# DOCUMENT: CONVERSION TO NUMPY AND USE OF self.value
#    • self.value is the lingua franca of (and always) the output of an executeMethod
#           Mechanisms:  value is always 2D np.array (to accomodate multiple states per Mechanism
#           All other Function objects: value is always 1D np.array
#    • Mechanism.value is always an indexible object of which the first item is a 1D np.array
#               (corresponding to the value of Mechanism.outputState and Mechanism.outputStates.items()[0]
#     Mechanism.variable[i] <-> Mechanism.inputState.items()e[i]
#     Mechanism.value[i] <-> Mechanism.ouptputState.items()e[i]
#    • variable = input, value = output
#    • Function.variable and Function.value are always converted to 2D np.ndarray
#             (never left as number, or 0D or 1D array)
#             [CLEAN UP CODE THAT TESTS FOR OTHERWISE] - this accomodate the possiblity of multiple states,
#                 each of which is represented by a 1D array in variable and value
#             Whenever self.value is set, should insure it is a 2D np.ndarray
#             output of projections should always be 1D array, since each corresponds to a single state
#     variable AND Mechanism output specification:
#     [0, 1, 2] (i.e., 1D array) => one value for the object
#                                (e.g., input state for a mapping projection, or param value for a ControlSignal projection)
#     [[0, 1, 2]] (i.e., 2D array) => multiple values for the objectn (e.g., states for a mechanism)
#     CONTEXTUALIZE BY # OF INPUT STATES:  IF ONLY ONE, THEN SPECIFY AS LIST OF NUMBERS;  IF MULITPLE, SPECIFIY EACH AS A LIST

# DOCUMENT: When "chaining" processes (such that the first Mechanism of one Process becomes the last Mechanism
#               of another), then that Mechanism loses its Mapping Projection from the input_state
#               of the first Process.  The principole here is that only "leaves" in a Process or System
#              (i.e., Mechanisms with otherwise unspecified inputs sources) get assigned Process.input_state Mappings
#
# DOCUMENT: UPDATE READ_ME REGARDING self.variable and self.value constraints
# DOCUMENT:  ADD "Execution" section to each description that explains what happens when object executes:
#                 - who calls it
#                 - relationship to update
#                 - what gets called
#
# DOCUMENT: ControlSignals are now NEVER specified for params by default;
#           they must be explicitly specified using ParamValueProjection tuple: (paramValue, kwControlSignal)
#     - Clean up ControlSignal InstanceAttributes
# DOCUMENT instantiate_mechanism_state_list() in Mechanism
# DOCUMENT: change comment in DDM re: EXECUTE_METHOD_RUN_TIME_PARAM
# DOCUMENT: Change to MechanismInputState, MechanismOutputState re: ownerMechanism vs. ownerValue
# DOCUMENT: use of runtime params, including:
#                  - specification of value (exposed or as tuple with ModulationOperation
#                  - role of  ExecuteMethodRuntimeParamsPref / ModulationOperation
# DOCUMENT: INSTANTIATION OF EACH DEFAULT ControlSignal CREATES A NEW outputState FOR DefaultController
#                                AND A NEW inputState TO GO WITH IT
#                                UPDATES VARIABLE OF ownerMechanism TO BE CORRECT LENGTH (FOR #IN/OUT STATES)
#                                NOTE THAT VARIABLE ALWAYS HAS EXTRA ITEM (I.E., ControlSignalChannels BEGIN AT INDEX 1)
# DOCUMENT: IN INSTANTIATION SEQUENCE:
#              HOW MULTIPLE INPUT AND OUTPUT STATES ARE HANDLED
#             HOW ITEMS OF variable AND ownerMechanism.value ARE REFERENCED
#             HOW "EXTERNAL" INSTANTIATION OF MechanismStates IS DONE (USING ControlSignal.instantiateSender AS E.G.)
#             ADD CALL TO Mechanism.update_value SEQUENCE LIST
# DOCUMENT: DefaultController
# DOCUMENT: Finish documenting def __init__'s
# DOCUMENT: (In Utility):
                        #     Instantiation:
                        #         A utility function can be instantiated in one of several ways:
                        # IMPLEMENTATION NOTE:  *** DOCUMENTATION
                        # IMPLEMENTATION NOTE:  ** DESCRIBE VARIABLE HERE AND HOW/WHY IT DIFFERS FROM PARAMETER
# DOCUMENT Runtime Params:
#              kwMechanismInputStateParams,
#              kwMechanismParameterStateParams,
#              kwMechanismOutputStateParams
#              kwProjectionParams
#              kwMappingParams
#              kwControlSignalParams
#              <projection name-specific> params
    # SORT OUT RUNTIME PARAMS PASSED IN BY MECHANISM:
    #    A - ONES FOR EXECUTE METHOD (AGGREGATION FUNCTION) OF inputState
    #    B - ONES FOR PROJECTIONS TO inputState
    #    C - ONES FOR EXECUTE METHOD (AGGREGATION FUNCTION) OF parmaterState
    #    D - ONES FOR EXECUTE METHOD OF MECHANISM - COMBINED WITH OUTPUT OF AGGREGATION FUNCTION AND paramsCurrent
    #    E - ONES FOR PROJECTIONS TO parameterState
    #  ?? RESTRICT THEM ONLY TO CATEGORY D
    #  ?? ALLOW DICT ENTRIES FOR EACH (WITH DEDICATED KEYS)
#
#endregion

#region PREFERENCES: ---------------------------------------------------------------------------------------------------------

# FIX:  SHOULD TEST FOR prefsList ABOVE AND GENERATE IF IT IS NOT THERE, THEN REMOVE TWO SETS OF CODE BELOW THAT DO IT
#
# FIX: Problem initializing classPreferences:
# - can't do it in class attribute declaration, since can't yet to refer to class as owner (since not yet instantiated)
# - can't use @property, since @setters don't work for class properties (problem with meta-classes or something)
# - can't do it by assigning a free-standing preference set, since its owner will remain DefaultProcessingMechanism
#     (this is not a problem for objects, since they use the @setter to reassign ownership)
# - another side effect of the problem is:
#   The following works, but changing the last line to "PreferenceLevel.CATEGORY" causes an error
#     DDM_prefs = FunctionPreferenceSet(
#                     # owner=DDM,
#                     prefs = {
#                         kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#                         kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                         kpExecuteMethodRuntimeParamsPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.CATEGORY)})

# FIX: SOLUTION TO ALL OF THE ABOVE:  CHANGE LOG PREF TO LIST OF KW ENTRIES RATHER THAN BOOL COMBOS (SEE LOG)
# FIX: Problems validating LogEntry / Enums:
# - Status:  commented out lines in PreferenceSet.validate_setting and PreferenceSet.validate_log:
# - Validating source of LogEntry setting (as being from same module in which PreferenceSet was declared)
#      is a problem for DefaultPreferenceSetOwner, since it can be assigned to objects declared in a different module
#      PROBLEM CODE (~660 in PreferenceSet.validate_log:
#             candidate_log_entry_value = candidate_log_class.__dict__['_member_map_'][log_entry_attribute]
#             global_log_entry_value = LogEntry.__dict__['_member_map_'][log_entry_attribute]
# - Validating that settings are actually enums is a problem when assigning boolean combinations
#      (which are not members of the enum)
#      PROBLEM CODE (~630 in PreferenceSet.validate_setting):
#             setting_OK = iscompatible(candidate_setting, reference_setting, **{kwCompatibilityType:Enum}
# ** FIX: # FIX: SHOULDN'T BE ABLE TO ASSIGN enum TO PREF THAT DOESN'T REQUIRE ONE (IN Test Script:)
#           my_DDM.prefs.verbosePref = LogEntry.TIME_STAMP
# IMPROVE: Modify iscompatible to distinguish between numbers and enums; then enforce difference in set_preference

# FIX: Add specification of setting type to pref @setter, that is passed to PreferenceSet.set_preference for validation

# FIX:  replace level setting?? (with one at lower level if setting is not found at level specified)
# QUESTION:  IS PreferenceSet.level attribute ever used?

# - implement: move defaults (e.g., defaultMechanism) to preferences

# - IMPLEMENT: change pref names from name_pref to namePref
#              (rectifying whatever conflict that will produce with other names)
#endregion

#region LOG: -----------------------------------------------------------------------------------------------------------------
#
# IMPLEMENT:
#             0) MOVE LIST OF RECORDED ENTRIES TO kwLogEntries param, AND USE logPref TO TURN RECORDING ON AND OFF
#             X) VALIDATE LOG VALUES (IN set_preferences)
#             Y) Fix CentralClock
#             4) IMPLEMENT RELEVANT SETTER METHODS IN Process, Mechanism and Projections (AKIN TO ONES IN MechanismState):
#                  MOVE IT TO LEVEL OF Function??
#             1) IMPLEMENT LOGGING "SWITCH" SOMEWHERE THAT TURNS LOGGING ON AND OFF: activate_logging, deactive_logging
#                 (PROCESS.configuration.prefs.logPref?? OR AT SYSTEM LEVEL?? OR AS PREF OR ATTRIBUTE FOR EVERY OBJECT?
#                IMPLEMENT THIS IN "IF" STATEMENT OF value SETTER METHODS
#                          MAKE SURE THIS CONTROLS APPENDING OF VALUES TO ENTRIES IN A CONTEXT-APPROPRIATE WAY
#             3) FINISH WORKING OUT INITIALIZATION (IN Function AND IN Log):
#                 SHOULD TRY TO GET ENTRIES FROM logPrefs??
#                 SHOULD USE classLogEntries AS DEFAULT IN CALL TO Log;
#                 SHOULD ADD SysetmLogEntries IN CALL TO Log (IN FUNCTIONS OR IN LOG??)
#                 SHOULD ADD kwLogEntries PARAM IN WHICH VARIABLES CAN BE ASSIGNED
#             5) DEAL WITH validate_log IN PreferenceSet
#             6) REVISIT Log METHODS (add_entry, log_entries, etc.) TO MAKE SURE THEY MAKE SENSE AND FUNCTION PROPERLY:
#                 ADAPT THEM TO LogEntry tuple FORMAT
#     WHEN DONE, SEARCH FOR FIX LOG:
#
# DOCUMENTATION: ADD DESCRIPTION OF HOW LOGGING IS TURNED ON AND OFF ONCE THAT IS IMPLEMENTED
#
# IMPLEMENT: ORDER OUTPUT ALPHABETICALLY (OR IN SOME OTHER CONSISTENT MANNER)
# IMPLEMENT: dict that maps names of entries (_iVar names) to user-friendly names
# IMPLEMENT: validate logPrefs in PrererenceSet.validate_log by checking list of entries against dict of owner object
# FIX: REPLACE ENUM/BOOLEAN BASED ENTRY SPECIFICATION WITH LIST/KW-BASED ONE
# Define general LogEntry enum class for common entry flags in Globals.Log
# Validate in PreferenceSet.__init__ that any setting being assigned to _log_pref:
# - is an enum that has all of the entries in Globals.LogEntry with the same values
# - is from the same module as the owner:  candidate_info.__module is Globals.LogEntry.__module__
# PROBLEM IS THAT CAN'T VALIDATE BOOLEAN COMBINATIONS WHICH ARE SIMPLE ints, NOT ENUM MEMBERS)
#
#endregion

#region DEFAULTS: ------------------------------------------------------------------------------------------------------------
#
# - IMPLEMENT DefaultControlMechanism(object) / DefaultController(name) / kwSystemDefaultController(str)
#
# - SystemDefaultInputState and SystemDefaultOutputState:
# - SystemDefaultSender = ProcessDefaultInput
# - SystemDefaultReceiver = ProcessDefaultOutput
# - DefaultController
#
# Reinstate Default mechanismState and projections and DefaultMechanism
# *** re-implement defaults:
#      In mechanismState() and Function:
#          DefaultMechanismState
#      In projection() and Function:
#          DefaultProjection
#      In Process_Base:
#          kwProcessDefaultProjection: Functions.Projections.Mapping
#          kwProcessDefaultProjectionFunction: Functions.Utility.LinearMatrix
#  DefaultMechanism is now being assigned in Process;
#  -  need to re-instate some form of set_default_mechanism() in Mechanism
#
#endregion

#region FULL SIMULATION RUN --------------------------------------------------------------------------------------------
#
# IMPLEMENT run Function (in Main.py):
#    1) Execute system and generate output
#    2) Call stimulus estimation/expectation mechanism update method to generate guess for next stimulus
#    3) Call EVC update estimation/expectation mechanism's guess as the System's input (self.variable),
#            and assign ControlSignals
#
# IMPLEMENT:  MAKE SURE THAT outputState.values REMAIN UNCHANGED UNTIL NEXT UPDATE OF MECHANISM
# TIMING VERSION:
#                             PHASE
# MECHANISMS              1     2    3
# ----------            ---------------
# Input                   X
# Reward                        X
# StimulusPrediction      X
# RewardPrediction              X
# DDM                     X          X
# Response                X
# EVC                                X
#
# PROCESSES
# ----------
# TaskExecution:      [(Input, 1), (DDM, 1)]
# RewardProcessing:   [(Reward, 2), (RewardPrediction, 2), (EVC, 3)]
# StimulusPrediction: [(Input, 1), (StimulusPrediction, 1), (DDM, 3), (EVC, 3)]
#
# FIX: NEED TO BE ABLE TO SPECIFY phaseSpec FOR EVC;  EITHER:
#       ALLOW EVC TO BE IN A PROCESS, AND RECEIVE PROCESS-SPECIFIED PROJECTIONS,
#       WHICH SHOULD AUTOMATICALLY INSTANTIATE CORRESPONDING MONITORED STATES (EVC.inputStates):
#       THAT IS:
#           WHEN A PROCESS INSTANTIATES AN PROJECTION TO AN EVC MECHANISM,
#           IT SHOULD NOT JUST ADD THE PROJECTION TO THE PRIMARY INPUT STATE
#           BUT RATHER CREATE A NEW inputState FOR IT (CURRENTLY ALL ARE ADDED TO THE PRIMARY inputState)
#      OR ADD phaseSpec FOR EACH inputState OF EVC (MAKE THIS A FEATURE OF ALL MECHANISMS?)
#
# FIX: AUTOMIATCALLY DETECT HOW MANY ROOTS (INPUTS THAT DON'T RECEIVE PROJECTIONS OTHER THAN FROM PROCESS)
#      len(System.input) == number of roots
# FIX: In sigmoidLayer:
#        "range" param is 2-item 1D array
#        executeMethod is LinearCombination (since it is a param) so executeMethod outPut is a single value
#        Need to suppress execute method, or assign some other one (e.g., CombineVectors)
#
# endregion

#region EVC ----------------------------------------------------------------------------------------------------------
#
# NOTE:  Can implement reward rate valuation by:
# - implementing reward mechanism (gets input from environment)
# - instantiating EVC with:
# params={
#     kwMonitoredOutputStates:[[reward_mechanism, DDM.outputStates[DDM_RT]],
#     kwExecuteMethodParams:{kwOperation:LinearCombination.Operation.PRODUCT,
#                            kwWeights:[1,1/x]}}
#    NEED TO IMPLEMENT 1/x NOTATION FOR WEIGHTS IN LinearCombination
#
# REFACTORING NEEDED:
# ? MODIFY MechanismState.instantiate_projections TO TAKE A LIST OF PROJECTIONS AS ITS ARG
# √ ADD METHOD TO Mechanism:  instantiate_projections:
#      default:  ADD PROJECTION TO (PRIMARY) inputState
#      optional arg:  inputState (REFERENCED BY NAME OR INDEX) TO RECEIVE PROJECTION,
#                     OR CREATE NEW inputState (INDEX = -1 OR NAME)
# ? MODIFY DefaultProcessingMechanism TO CALL NEW METHOD FROM instantiate_control_signal_channel
# - FIX: ?? For SystemControlMechanism (and subclasses) what should default_input_value (~= variable) be used for?
# - EVC: USE THE NEW METHOD TO CREATE MONITORING CHANNELS WHEN PROJECIONS ARE AUTOMATCIALLY ADDED BY A PROCESS
#         OR IF params[kwInputStates] IS SPECIFIED IN __init__()
#
# - IMPLEMENT: EXAMINE MECHANISMS (OR OUTPUT STATES) IN SYSTEM FOR monitor ATTRIBUTE,
#                AND ASSIGN THOSE AS MONITORED STATES IN EVC (inputStates)
#
# - IMPLEMENT: .add_projection(Mechanism or MechanismState) method:
#                   - add controlSignal projection from EVC to specified Mechanism/MechanismState
#                   - validate that Mechanism / MechanismState.ownerMechanism is in self.system
#                   ?? use Mechanism.add_projection method
# - IMPLEMENT: kwExecuteMethodParams for cost:  operation (additive or multiplicative), weight?
# - TEST, DOCUMENT: Option to save all EVC policies and associated values or just max
# - IMPLEMENT: Control Mechanism that is assigned as default with kwSystem specification
#               ONCE THAT IS DONE, THEN FIX: IN System.instantiate_attributes_before_execute_method:
#                                                         self.controller = EVCMechanism(params={kwSystem: self})#
# - IMPLEMENT: ??execute_system method, that calls execute.update with input pass to System at run time?
# ? IMPLEMENT .add_projection(Mechanism or MechanismState) method that adds controlSignal projection
#                   validate that Mechanism / MechanismState.ownerMechanism is in self.system
#                   ? use Mechanism.add_projection method
# - IMPLEMENT: kwMonitoredOutputStatesOption for individual Mechanisms (in SystemControlMechanism):
#        TBI: Implement either:  (Mechanism, MonitoredOutputStatesOption) tuple in kwMonitoredOutputStates specification
#                                and/or kwMonitoredOutputStates in Mechanism.params[]
#                                         (that is checked when ControlMechanism is implemented
#        DOCUMENT: if it appears in a tuple with a Mechanism, or in the Mechamism's params list,
#                      it is applied to just that mechanism
#
# IMPLEMENT: call SystemControlMechanism should call ControlSignal.instantiate_sender()
#                to instantaite new outputStates and Projections in take_over_as_default_controller()
#
# IMPLEMENT: kwPredictionInputTarget option to specify which mechanism the EVC should use to receive, as input,
#                the output of a specified prediction mechanims:  tuple(PredictionMechanism, TargetInputMechanism)
#
# IMPLEMENT: EVCMechanism.MonitoredOutputStates (list of each Mechanism.outputState being monitored)
# DOCUMENT: DIFFERENCES BETWEEN EVCMechanism.inputStates (that receive projections from monitored States) and
#                               EVCMechanism.MonitoredOutputStates (the terminal states themselves)

# FIX: CURRENTLY DefaultController IS ASSIGNED AS DEFAULT SENDER FOR ALL CONTROL SIGNAL PROJECTIONS IN
# FIX:                   ControlSignal.paramClassDefaults[kwProjectionSender]
# FIX:   SHOULD THIS BE REPLACED BY EVC?
# FIX:  CURRENTLY, kwCostAggregationFunction and kwCostApplicationFunction ARE SPECIFIED AS INSTANTIATED FUNCTIONS
#           (IN CONTRAST TO executeMethod  WHICH IS SPECIFIED AS A CLASS REFERENCE)
#           COULD SWITCH TO SPECIFICATION BY CLASS REFERENCE, BUT THEN WOULD NEED
#             CostAggregationFunctionParams and CostApplicationFunctionParams (AKIN TO executeMethodParams)
#
# FIX: self.variable:
#      - MAKE SURE self.variable IS CONSISTENT WITH 2D np.array OF values FOR kwMonitoredOutputStates
#
# DOCUMENT:  protocol for assigning DefaultControlMechanism
#           Initial assignment is to SystemDefaultCcontroller
#           When any other SystemControlMechanism is instantiated, if params[kwMakeDefaultController] = True
#                then the class's take_over_as_default_controller() method
#                     is called in instantiate_attributes_after_execute_method
# it moves all ControlSignal Projections from DefaultController to itself
#
# FIX: IN ControlSignal.instantiate_sender:
# FIX 6/28/16:  IF CLASS IS SystemControlMechanism SHOULD ONLY IMPLEMENT ONCE;  THEREAFTER, SHOULD USE EXISTING ONE
#
# FIX: SystemControlMechanism.take_over_as_default_controller() IS NOT FULLY DELETING DefaultController.outputStates
#
# FIX: PROBLEM - SystemControlMechanism.take_over_as_default_controller()
# FIX:           NOT SETTING sendsToProjections IN NEW CONTROLLER (e.g., EVC)
#
# SOLUTIONS:
# 1) CLEANER: use instantiate_sender on ControlSignal to instantiate both outputState and projection
# 2) EASIER: add self.sendsToProjections.append() statement in take_over_as_default_controller()


# BACKGROUND INFO:
# instantiate_sender normally called from Projection in instantiate_attributes_before_execute_method
#      calls sendsToProjection.append
# instantiate_control_signal_projection normally called from ControlSignal in instantiate_sender
#
# Instantiate EVC:  __init__ / instantiate_attributes_after_execute_method:
#     take_over_as_default(): [SystemControlMechanism]
#         iterate through old controller’s outputStates
#             instantiate_control_signal_projection() for current controller
#                 instantiate_mechanism_state() [Mechanism]
#                     state_type() [MechanismOutputState]


#endregion

#region SYSTEM ---------------------------------------------------------------------------------------------------------
#
# System module:
#
# TOOLS
#             Visualizer
#               Cytoscape
#               Gephi
#
#             Directed acyclic graph
#             Topological sort
#
#             Methods for develping hierarchical models
#             Stan
#             Winbug
#
#             Python networkx:
#             https://networkx.github.io/
#
#
#    PARSER:
#    Specify:
#      tuples: (senderMech, receiverMech, [projection])
#                     senderMech & receiverMech must be mechanism specifications
#                     projectionMatrix must specify either:
#                          + Mapping projection object
#                          + kwIdentityMatrix: len(sender.value) == len(receiver.variable)
#                          + kwFull (full cross-connectivity) [** ADD THIS AS SPEC FOR LinearMatrix FUNCTION)
#                          + timing params
#      Processes (and use their configurations)
#    Run toposort to get linear structure
#
#    EXECUTION:
#    run function:
#        Calls each Process once per time step (update cycle)
#
#    "SEQUENTIAL"/"ANALYTIC" MODE:
#    1) Call every Process on each cycle
#        a) Each Process calls the Mechanisms in its Configuration list in the sequence in which they appear;
#            the next one is called when Mechanism.receivesFromProjections.frequency modulo CurrentTime() = 0
#
# VS:
#        a) Each Process polls all the Mechanisms in its Configuration list on each cycle
#            each one is called when Mechanism.receivesFromProjections.frequency modulo CurrentTime() = 0
#
# SEQUENTIAL MODE:
#     COMPUTE LCD (??GCF??)
#     STEP THROUGH 0->LCD
#     EACH FIRES WHEN ITS FREQ == STEP #
#
# CASCADE MODE:
#     EVERY MECHANISM UPDATES EVERY STEP
#     BUT IT DOES SO WITH SCALE = 1/FREQ
#
# Update Cycle:
#     Each Process calls update method of each mechanism in its configuration
#     Mechanisms are called in reverse order so that updating is synchronous
#         (i.e., updating of each mechanism is indpendent of the influence of any others in a feed-forward chain)
#     Each mechanism:
#         - updates its inputState(s) by calling its projection(s)
#             - projections are updated and values included in inputState update based on projection's timing param
#         - runs its execute method
#         - updates its outputState(s)
#
# TimeScale params:
#     Number of timesteps per trial (default:  100??)
#
# ProjectionTiming params [ProjectionTiming namedtuple: (phase, frequency, scale)]
#     Phase (time_steps): determines when update function starts relative to start of run
#     Frequency (int [>0] or float [0-1]): cycle time for projection's contribution to updating of inputState.variable:
#                               - if int, updates every <int> time_steps
#                               - if float, multiplied times self.value and used to update every cycle
#     Scale (float [0-1]):  scales projection's contribution to inputState.variable
#                            (equivalent to rate constant of time-averaged net input)
#     Function (UpdateMode):  determines the shape of the cycling function;
#                             - default is delta function:  updates occur only on time_steps modulo frequency
#                             - future versions should add other functions
#                               (e.g,. square waves and continuous function to "temporally smooth" update function
# LATEST VERSION:
#   - phaseSpec for each Mechanism in Process::
#        integers:
#            specify time_step (phase) on which mechanism is updated (when modulo time_step == 0)
#                - mechanism is fully updated on each such cycle
#                - full cycle of System is largest phaseSpec value
#        floats:
#            values to the left of the decimal point specify the "cascade rate":
#                the fraction of the outputvalue used as the input to any projections on each (and every) time_step
#            values to the right of the decimal point specify the time_step (phase) at which updating begins

#
# QUESTION: SHOULD OFF PHASE INPUT VALUES BE SET TO EMPTY OR NONE INSTEAD OF 0?
#           IN SCRIPTS AND EVCMechanism.get_simulation_system_inputs()
# FIX: Replace toposort with NetworkX: http://networkx.readthedocs.io/en/stable/reference/introduction.html
# IMPLEMENT: Change current System class to ControlledSystem subclass of System_Base,
#                   and purge System_Base class of any references to or dependencies on controller-related stuff
# IMPLEMENT: MechanismTuple class for mech_tuples: (mechanism, runtime_params, phase)
#            (?? does this means that references to these in scripts will require MechanismTuple declaration?)
# IMPLEMENT: *** ADD System.controller to execution_list and
#                execute based on that, rather than dedicated line in System.execute
# IMPLEMENT: *** sort System.execution_list (per System.inspect() and exeucte based on that, rather than checking modulos
# IMPLEMENT: *** EXAMINE MECHANISMS (OR OUTPUT STATES) IN SYSTEM FOR monitor ATTRIBUTE,
#                AND ASSIGN THOSE AS MONITORED STATES IN EVC (inputStates)
# IMPLEMENT: System.execute() should call EVC.update or EVC.execute_system METHOD??? (with input passed to System on command line)
# IMPLEMENT: Store input passed on command line (i.e., at runtime) in self.input attribute (for access by EVC)??
# IMPLEMENT: run() function (in Systems module) that runs default System
# IMPLEMENT: System.inputs - MAKE THIS A CONVENIENCE LIST LIKE System.terminalMechanisms
# IMPLEMENT: System.outputs - MAKE THIS A CONVENIENCE LIST LIKE System.terminalMechanisms
#
# FIX: NOTES: MAKE SURE System.execute DOESN'T CALL EVC FOR EXECUTION (WHICH WILL RESULT IN INFINITE RECURSION)
#
# FIX: NEED TO INSURE THAT self.variable, self.inputs ARE 3D np.arrays (ONE 2D ARRAY FOR EACH PROCESS IN kwProcesses)
# FIX:     RESTORE "# # MODIFIED 6/26/16 NEW:" IN self.validate_variable
# FIX:     MAKE CORRESPONDING ADJUSTMENTS IN self.instantiate_execute_method (SEE FIX)
#
# FIX: Output of default System() produces two empty lists
#
#endregion

#region FUNCTIONS: -----------------------------------------------------------------------------------------------------------
#
#  validate_execute_method:
#
# DOCUMENT:
#    - Clean up documentation at top of module
#    - change relevant references to "function" to "execute method"
#    - note that run time params must be in kwExecuteMethodParams
#    - Add the following somewhere (if it is not already there):
#     Parameters:
#         + Parameters can be assigned and/or changed individually or in sets, by:
#             including them in the initialization call (see above)
#             calling the adjust method (which changes the default values for that instance)
#             including them in a call the execute method (which changes their values for just for that call)
#         + Parameters must be specified in a params dictionary:
#             the key for each entry should be the name of the parameter (used to name its state and projections)
#             the value for each entry can be a:
#                 + MechanismState object: it will be validated
#                 + MechanismState subclass: a default state will be implemented using a default value
#                 + dict with specifications for a MechanismState object to create:
#                 + ParamValueProjection tuple: a state will be implemented using the value and assigned the projection
#                 + projection object or class: a default state will be implemented and assigned the projection
#                 + value: a default state will be implemented using the value

# IMPLEMENT: MODIFY SO THAT self.execute (IF IT IS IMPLEMENTED) TAKES PRECEDENCE OVER kwExecuteMethod
#                 BUT CALLS IT BY DEFAULT);  EXAMPLE:  AdaptiveIntegratorMechanism
# IMPLEMENT:  change specification of params[kwExecuteMethod] from class to instance (as in ControlSignal functions)
# IMPLEMENT:  change validate_variable (and all overrides of it) to:
#              validate_variable(request_value, target_value, context)
#              to parallel validate_params, and then:

# IMPLEMENT: some mechanism to disable instantiating MechanismParameterStates for parameters of an executeMethod
#                that are specified in the script
#            (e.g., for EVC.executeMethod:
#                - uses LinearCombination,
#                - want to be able to specify the parameters for it
#                - but do not need any parameterStates assigned to those parameters
#            PROBLEMS:
#                - specifying parameters invokes instantation of parameterStates
#                    (note: can avoid parameterState instantation by not specifying parameters)
#                - each parameterState gets assigned its own executeMethods, with the parameter as its variable
#                - the default executeMethod for a parameterState is LinearCombination (using kwIdentityMatrix)
#                - that now gets its own parameters as its variables (one for each parameterState)
#                - it can't handle kwOperaton (one of its parameters) as its variable!
#            SOLUTION:
#                - kwExecuteMethodParams: {kwMechanismParameterState: None}}:  suppresses MechanismParameterStates
#                - handled in Mechanism.instantiate_execute_method_parameter_states()
#                - add DOCUMENTATION in Functions and/or Mechanisms or MechanismParameterStates;
#                      include note that executeMethodParams are still accessible in paramsCurrent[executeMethodParams]
#                      there are just not any parameterStates instantiated for them
#                          (i.e., can't be controlled by projections, etc.)
#                - TBI: implement instantiation of any specs for parameter states provided in kwMechanismParameterStates
#
# Implement: recursive checking of types in validate_params;
# Implement: type lists in paramClassDefaults (akin requiredClassParams) and use in validate_params
            # IMPLEMENTATION NOTE:
            #    - currently no checking of compatibility for entries in embedded dicts
            #    - add once paramClassDefaults includes type lists (as per requiredClassParams)
# Implement categories of Utility functions using ABC:
# - put checks for constraints on them (e.g., input format = output format)
# - associate projection and mechanismState categories with utility function categories:
#    e.g.:  mapping = transform;  input & output states = aggregate
#
#endregion

#region PROCESS: -------------------------------------------------------------------------------------------------------------
#
# - DOCUMENT: Finish editing Description:
#             UPDATE TO INCLUDE Mechanism, Projection, Mechanism FORMAT, AND (Mechanism, Cycle) TUPLE
#
# - FIX: NEED TO DEAL WITH SITUATION IN WHICH THE SAME MECHANISM IS USED AS THE FIRST ONE IN TWO DIFFERENT PROCESSES:
#        ?? WHAT SHOULD BE ITS INPUT FROM THE PROCESS:
#           - CURRENTLY, IT GETS ITS INPUT FROM THE FIRST PROCESS IN WHICH IT APPEARS
#           - IMPLEMENT: ABILITY TO SPECIFY WHICH PROCESS(ES?) CAN PROVIDE IT INPUT
#                        POSSIBLY MAP INPUTS FROM DIFFERENT PROCESSES TO DIFFERENT INPUT STATES??
#
# - IMPLEMENT: Autolink for configuration:
#               WHAT TO DO WITH MECHANISMS THAT RECEIVE A PROJECTION W/IN THE LIST BUT NOT THE PRECEDING
#               OVERRIDE MODE:  serial projections only within the config list
#               INHERIT MODE:   mechanisms retain all pre-specified projections:
#                                  ?? check for orphaned projections? mechanisms in NO process config??
#
# - fix: how to handle "command line" execute method parameters (i.e., specified in config tuple):
#        in check args they get incorporated into paramsCurrent, but into parameterState.value's
#        combining all of them in mechanism execute method would be "double-counting"
#        - only count the ones that changed?
#        - handle "command line" params separately from regular ones (i.e., isolate in check_args)??
#        - pass them through parameterState execute function
#              (i.e., pass them to parameterState.execute variable or projection's sender??)
# - implement:
#     • coordinate execution of multiple processes (in particular, mechanisms that appear in more than one process)
#     • deal with different time scales
#     • response completion criterion (for REAL_TIME mode) + accuracy function
#     • include settings and log (as in ControlSignal)
#
# - implement:  add configuration arg to call, so can be called with a config
#
# - implement: alias Process_Base to Process for calls in scripts
#
#
# *** DECIDE HOW TO HANDLE RUNNING OF ALL execute FUNCTIONS ON INIT OF OBJECT:
#    ?? DO IT BUT SUPPRESS OUTPUT?
#    ?? SHOW OUTPUT BUT FLAG AS INITIALIZATION RUN
#    ?? USE CONTEXT TO CONDUCT ABBREVIATED RUN??
#
# execute methods: test for kwSeparator+kwFunctionInit in context:
#          limit what is implemented and/or reported on init (vs. actual run)
#endregion

#region MECHANISM: -----------------------------------------------------------------------------------------------------------
#
#
# CONFIRM: VALIDATION METHODS CHECK THE FOLLOWING CONSTRAINT: (AND ADD TO CONSTRAINT DOCUMENTATION):
# DOCUMENT: #OF OUTPUTSTATES MUST MATCH #ITEMS IN OUTPUT OF EXECUTE METHOD **
#
# IMPLEMENT / DOCUMENT 7/20/16: kwExecuteMethodOutputStateValueMapping (dict) and attendant param_validation:
#            required if self.execute is not implemented and return value of kwExecuteMethod is len > 1
#
# IMPLEMENT: 7/3/16 inputValue (== self.variable) WHICH IS 2D NP.ARRAY OF inputState.value FOR ALL inputStates
# FIX: IN instantiate_mechanismState:
# FIX: - check that constraint_values IS NOW ONLY EVER A SINGLE VALUE
# FIX:  CHANGE ITS NAME TO constraint_value
# Search & Replace: constraint_values -> constraint_value
#
# - Clean up Documentation
# - add settings and log (as in ControlSignal)
# - Fix: name arg in init__() is ignored
#
# - MODIFY add_projection
#         IMPLEMENTATION NOTE:  ADD FULL SET OF MechanismParameterState SPECIFICATIONS
#
# IMPLEMENT: EXAMINE MECHANISMS (OR OUTPUT STATES) IN SYSTEM FOR monitor ATTRIBUTE,
#                AND ASSIGN THOSE AS MONITORED STATES IN EVC (inputStates)
#
# - IMPLEMENT: CLEAN UP ORGANIZATION OF STATES AND PARAMS
# Mechanism components:                Params:
#   MechanismInputStates      <- MechanismInputStateParams
#   MechanismParameterStates  <- MechanismParameterStateParams (e.g., Control Signal execute method)
#   MechanismOutputStates     <- MechanismOutputStateParams
#   self.execute              <- MechanismExecuteMethod, MechanismExecuteMethodParams (e.g., automatic drift rate)
#
# IMPLEMENT:  self.execute as @property, which can point either to _execute or paramsCurrent[kwExecuteMethod]
#
# - IMPLEMENTATION OF MULTIPLE INPUT AND OUTPUT STATES:
# - IMPLEMENT:  ABSTRACT HANDLING OF MULTIPLE STATES (AT LEAST FOR INPUT AND OUTPUT STATES, AND POSSIBLE PARAMETER??
# - Implement: Add MechanismStateSpec tuple specificaton in list for  kwMechanismInputState and MechanismOUtputStates
#        - akin to ParamValueProjection
#        - this is because OrderedDict is a specialty class so don't want to impose their use on user specification
#        - adjust validate_params and instantiate_output_state accordingly
# - Implement: allow list of names, that will be used to instantiate states using self.value
# - Implement: allow dict entry values to be types (that should be checked against self.value)
#
# - NEED TO INITIALIZE:            kwMechanismStateValue: NotImplemented,
# - IMPLEMENTATION NOTE: move defaultMechanism to a preference (in Mechanism.__init__() or Process.__init())
# - IMPLEMENTATION NOTE: *** SHOULD THIS UPDATE AFFECTED PARAM(S) BY CALLING RELEVANT PROJECTIONS?
# -    ASSGIGN  *** HANDLE SAME AS MECHANISM STATE AND PROJECTION STATE DEFAULTS:
#                   create class level property:  inputStateDefault, and assign it at subclass level??
# - replace "state" with "mechanism_state"
# - Generalize validate_params to go through all params, reading from each its type (from a registry),
#                            and calling on corresponding subclass to get default values (if param not found)
#                            (as kwProjectionType and kwProjectionSender are currently handled)
# IN MECHANISMS validate_execute_method:
#   ENFORCEMENT OF CONSTRAINTS
#
# - Break out separate execute methods for different TimeScales and manage them in Mechanism.update_and_execute
#
# # IMPLEMENTATION NOTE: *** SHOULDN'T IT BE INSTANTIATED? (SEE BELOW)  (ARE SOME CONTEXTS EXPECTING IT TO BE A CLASS??)
#                          SEARCH FOR [kwExecuteFunction] TO SEE
#
# ??ADD self.valueType = type(func(inspect.getargspec(func).args))
#             assign this Function.__init__

#
# In instantiate_mechanism_state (re: 2-item tuple and Projection cases):
        # IMPLEMENTATION NOTE:
        #    - need to do some checking on state_spec[1] to see if it is a projection
        #      since it could just be a numeric tuple used for the variable of a mechanismState;
        #      could check string against ProjectionRegistry (as done in parse_projection_ref in MechanismState)
    # IMPLEMENTATION NOTE:
    #    - should create list of valid projection keywords and limit validation below to that (instead of just str)
#
# - implement:
#     Regarding ProcessDefaultMechanism (currently defined as Mechanism_Base.defaultMechanism)
#        # IMPLEMENTATION NOTE: move this to a preference (in Process??)
#        defaultMechanism = kwDDM
#
#endregion

#region MECHANISM_STATE: -----------------------------------------------------------------------------------------------------
#
# IMPLEMENT outputStateParams dict;  SEARCH FOR: [TBI + kwMechanismOutputStateParams: dict]
#
# *** NEED TO IMPLEMENT THIS (in MechanismState, below):
# IMPLEMENTATION NOTE:  This is where a default projection would be implemented
#                       if params = NotImplemented or there is no param[kwMechanismStateProjections]
#
# **** IMPLEMENTATION NOTE: ***
#                 FOR MechainismInputState SET self.value = self.variable of ownerMechanism
#                 FOR MechanismiOuptuState, SET variableClassDefault = self.value of ownerMechanism
#
# - MechanismState, ControlSignal and Mapping:
# - if "senderValue" is in **args dict, assign to variable in init
# - clean up documentation
#
         # - %%% MOVE TO MechanismState
         #  - MOVE kwMechanismStateProjections out of kwMechanismStateParams:
         #        # IMPLEMENTATION NOTE:  MOVE THIS OUT OF kwMechanismStateParams IF CHANGE IS MADE IN MechanismState
         #        #                       MODIFY KEYWORDS IF NEEDED
         #    and process in __init__ (instantiate_projections()) rather than in validate_params
         # - if so, then correct in instantiate_execute_method_params under Mechanism
         # - ADD instantiate_projection akin to instantiate_state in Mechanism
         # - ADD validate_projection() to subclass, that checks projection type is OK for mechanismState
#
## ******* MOVE THIS TO MechanismState
#                 try:
#                     from Functions.Projections.Projection import ProjectionRegistry
#                     projection_type = ProjectionRegistry[param_value.projection].subclass
#                 except ValueError:
#                     raise MechanismError("{0} not recognized as reference to a projection or projection type".
#                                          format(param_value.projection))
#
# ADD HANDLING OF PROJECTION SPECIFICATIONS (IN kwMechanismStateProjection) IN MechanismState SUBCLASSES
#                  MUST BE INCLUDED IN kwMechanismStateParams
#
# GET CONSTRAINTS RIGHT:
#    self.value === Mechanism.function.variable
#    self.value ===  MechanismOutputState.variable
#    Mechanism.params[param_value] === MechanismParameterState.value = .variable
#
    # value (variable) == owner's functionOutputValue since that is where it gets it's value
    #    -- ?? should also do this in Mechanism, as per inputState:
                # See:
                # Validate self.inputState.value against variable for kwExecuteMethod
                # Note:  this is done when inputState is first assigned,
                #        but needs to be done here in case kwExecuteMethod is changed
    # uses MappingProjetion as default projection
    # implement Aritmetic ADD Combination Function as kwExecuteMethod
    # implement default states (for use as default sender and receiver in Projections)

# *********************************************
# ?? CHECK FOR PRESENCE OF self.execute.variable IN Function.__init__ (WHERE self.execute IS ASSIGNED)
# IN MechanismOutputState:
#   IMPLEMENTATION NOTE: *** MAKE SURE self.value OF MechanismsOutputState.ownerMechanism IS
#                           SET BEFORE validate_params of MechanismsOutputState
# *********************************************
#
# FOR inputState:
#      self.value does NOT need to match variable of inputState.function
#      self.value MUST match self.param[kwExecutMethodOuptputDefault]
#      self.value MUST match ownerMechanisms.variable
#
#
# # IMPLEMENTATION NOTE:  *** SHOULD THIS ONLY BE TRUE OF MechanismInputState??
#         # If ownerMechanism is defined, set variableClassDefault to be same as ownerMechanism
#         #    since variable = self.value for MechanismInputState
#         #    must be compatible with variable for ownerMechanism
#         if self.ownerMechanism != NotImplemented:
#             self.variableClassDefault = self.ownerMechanism.variableClassDefault
#
#endregion

#region PROJECTION: ----------------------------------------------------------------------------------------------------------
#
# - IMPLEMENT:  WHEN ABC IS IMPLEMENTED, IT SHOULD INSIST THAT SUBCLASSES IMPLEMENT instantiate_receiver
#               (AS ControlSignal AND Mapping BOTH DO) TO HANDLE SITUATION IN WHICH MECHANISM IS SPECIFIED AS RECEIVER
# - FIX:  Move marked section of instantiate_projections(), check_projection_receiver(), and parse_projection_ref
#   FIX:      all to Projection_Base.__init__()
# - add kwFull to specification, and as default for non-square matrices
# - IMPLEMENTATION NOTE:  *** NEED TO SPECIFY TYPE OF MECHANIMSM_STATE HERE:  SHOULD BE DETERMINABLE FROM self.Sender
# - Implement generic paramProjection subclass of Projection:
#       stripped down version of ControlSignal, that has free-floating default inputState
#       used to control execute method params on a trial-by-trial basis (akin to use of tuples in configuration)
# - Fix: name arg in init__() is ignored
#
#endregion

#region MAPPING: ------------------------------------------------------------------------------------------------------
#
# DOCUMENT:
# If the sender outputState and/or the receiver inputState are not specified:
#    - a mapping will be created for only sender.outputState and receiver inputState (i.e., first state of each)
#    - the length of value for these states must match
#
#endregion

#region CONTROL_SIGNAL: ------------------------------------------------------------------------------------------------------
#
#      controlModulatedParamValues
#
# 0) MAKE SURE THAT kwProjectionSenderValue IS NOT PARSED AS PARAMS
#      NEEDING THEIR OWN PROJECTIONS (HOW ARE THEY HANDLED IN PROJECTIONS?) -- ARE THEWE EVEN USED??
#      IF NOT, WHERE ARE DEFAULTS SET??
# 2) Handle assignment of default ControlSignal sender (DefaultController)
#
# FIX ************************************************
# FIX: controlSignal prefs not getting assigned

# Fix: rewrite this all with @property:
#
# IMPLEMENT: when instantiating a ControlSignal:
#                   include kwDefaultController as param for assigning sender to DefaultController
#                   if it is not otherwise specified
#
#  IMPLEMENT option to add dedicated outputState for ControlSignal projection??
#
#
# IMPLEMENTATION NOTE:  ADD DESCRIPTION OF ControlSignal CHANNELS:  ADDED TO ANY SENDER OF A ControlSignal Projection:
    # USED, AT A MININUM, FOR ALIGNING VALIDATION OF inputStates WITH ITEMS IN variable
    #                      ?? AND SAME FOR FOR outputStates WITH value
    # SHOULD BE INCLUDED IN INSTANTIATION OF CONTROL MECHANISM (per SYSTEM DEFAULT CONTROL MECHANISM)
    #     IN OVERRIDES OF validate_variable AND
    #     ?? WHEREVER variable OF outputState IS VALIDATED AGAINST value (search for FIX)
#
#endregion

#region LEARNING: ------------------------------------------------------------------------------------------------------

# Two object types:
# 1) Comparator Mechanism:
#     - has two inputStates:  i) system output;  ii) training input
#     - computes some objective function on them (default:  Hadamard difference)
# 2) Training Projection:
#     - sender:  output of Comparator Mechanism
#     - receiver: Mapping Projection parameterState (or some equivalent thereof)
# Need to add parameterState to Projection class;  composition options:
#    - use MechanismParameterState
#    - extract core functionality from MechanismParameterState:
#        make it an object of its own
#        MechanismParameterState and Training Projection both call that object

# Projection mechanism:
# Generalized delta rule:
# weight = weight + (learningRate * errorDerivative * tranferDerivative * sampleSender)
# for sumSquared error function:  errorDerivative = (target - sample)
# for logistic activation function: transferDerivative = sample * (1-sample)
# NEEDS:
# - errorDerivative:  get from kwExecuteMethod of Comparator Mechanism
# - transferDerivative:  get from kwExecuteMethod of Process Processing Mechanism

#endregion

#region DDM_MECH: ------------------------------------------------------------------------------------------------------
#
# - Fix: combine paramsCurrent with executeParameterState.values, or use them instead??
# - Fix:  move kwDDM_AnalyticSolution back to kwExecuteMethodParams and adjust validation to allow non-numeric value
# - implement: add options to multiply or fully override parameterState.values
# - implement time_step and terminate()
# -  Clean up control signal params, modulation function, etc.
#        1) value field is initialized with self.value
#        2) value points to mechanism.outputState.value
#        3) params field is populated with list of params from paramsCurrent that are MechanismStateParams
#        4) duration field is updated at each time step or given -1
#    Make sure paramCurrent[<kwDDMparam>] IS BEING PROPERLY UPDATED (IN PROCESS?  OR MECHANISM?) BEFORE BEING USED
#                            (WHAT TOOK THE PLACE OF get_control_modulated_param_values)
# IMPLEMENT: ADD PARAM TO DDM (AKIN TO kwDDM_AnayticSolution) THAT SPECIFIES PRIMARY INPUTSTATE (i.e., DRIFT_RATE, BIAS, THRSHOLD)
#
#endregion

#region UTILITY: -------------------------------------------------------------------------------------------------------------
#
# Implement name arg to individual functions, and manage in __init__()
# Implement abstract Types (aggregate, transfer, tranform, objective)
# Implement subtypes of above
# Implement:  shortcircuit LinearCombination and Linear and LinearMatrix if params => identity
# LinearMatrix:
#   IMPLEMENTATION NOTE: Consider using functionOutputTypeConversion here
#   FIX:  IMPLEMENT BOTH kwFullConnectivityMatrix AND 2D np.array AND np.matrix OBJECTS
#
# IMPLEMENT:
#     IN LinearCombination kwWeights PARAM:  */x notation:
#         Signifies that item to which weight coefficient applies should be in the denominator of the product:
#         Useful when multiplying vector with another one, to divide by the specified element (e.g., in calcuating rates)
#      EXAMPLE:
#           kwWeights = [1, 1/x]   [1, 2/x]
#           variable =  [2, 100]   [2, 100]
#           result:     [2, .01]   [2, 0.2]
#
# IMPLEMENT: simple Combine() or Reduce() function that either sums or multiples all elements in a 1D array
# IMPLEMENT:  REPLACE INDIVIDUAL FUNCTIONS WITH ABILITY TO PASS REFERENCE TO NP FUNCTIONS (OR CREATE ONE THAT ALLOWS THIS)

#endregion


