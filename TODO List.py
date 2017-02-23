# NEW COMMENT
# **************************************************  ToDo *************************************************************
#region CURRENT: -------------------------------------------------------------------------------------------------------

# FIX:
#   MAKE SURE THAT WHEREVER variableClassDefaults OR paramClassDefaults ARE CHANGED IT IS LEGIT
#             I.E., THAT THIS BE OK FOR ALL OTHER INSTANCES OF THAT CLASS
#             FOR EXAMPLE, IN assign_params_to_dicts, WHERE A DEFAULT IS SPECIFIED IN THE ARG RATHER THAN classDefaults
# FIX:
#    0) Deal with function parameter assignment in update() of ParameterState
#        - move assignment of function params (Lines 714 and 742 in ParameterState)
#               into @property for value (Line 756) [DEBUG CRASH]
#        - assign params for function to owner in _instantiate_parameter_states (currently in assign_params_to_dicts??)
#          ?? use user_params from function?
#    1) Once function param assignment is fixed, add test that it is working to jenkins suite
#          (i.e., that assigning a value to the attribute for the parameter on the object (e.g., mechanism)
#                 changes its value for the Function
#    2) Add learning rate param (including global default)
#    3) For system vs. process learning:
#           Figure out why calling update_state for the matrix ParameterState works,
#                      but executing the LearningProjection to it does not
#    4) ObjectiveMechanisms:  MODIFY TO:
#                                d) Revise EVCMechainism._get_monitored_states() to NOT direclty assign weights
#                                           and exponents, but rather assign
#                                e) Document monitored_values and default_input_value (sets size of inputSTates)
#                                    (see RE-WRITE TO INDICATE:  (SEE ATTRIBUTE DESCRIPTION FOR monitored_values)
#                                f) parse MonitoredOUtputStates specification for monitored_values arg
#                                g) Fix EVC use of OBjectiveMechanism (needs to now call for Mapping Projection
#                                h) Accomodate WeightedError in OjbectiveMechanism using standard LinearComb function:
#                                            Matrix - IDENTITY MATRIX
#                                            Derivative - Linear
#          them where the ObjectiveMechanism is created (in its LinearFunction)
#     5) Purge DefaultMonitoringMechanism
#     6) ??Bother to make Comparator sublcass of ObjectiveMechanism
#                (that names its inputStates and creates the relevant set of outputStates -- see LearningProjection)
#     7) DDM weights for EVC mechanism:  Handle better in ObjectiveMechanism

# DOCUMENT:  Projection (vs. Mechanism):  single input/oputput, and single parameter;  no execution_id
#
# FIX: PUT ERROR HERE IF EVC AND/OR EVC_MAX ARE EMPTY (E.G., WHEN EXECUTION_ID IS WRONG)
#                 if EVC == EVC_max: (LINE 289 IN EVCAuxilliary)

# FIX: execution_token asynchrony:
#    * Since learning and controller execute after processing mechanisms:
#      - on the first pass, they ignore learning and control projections (since their excxecution_tokens == None
#      - on subsequent passes, they have the new (current) execution_token, while learning and control mechanisms
#             still have the last one
#      SOLUTION:  HAVE LEARNING AND CONTROL MECHANISMS GET THEIR execution_tokens FROM THEIR SYSTEM?
#                 OR HAVE SYSTEM ASSIGN ITS LEARNING AND CONTROL MECHANISMS THE CURRENT execution_token??
#      SOLUTION:  ASSIGN execution_token TO ALL MECHANISMS IN SYSTEM GRAPH AND LEARNING GRAPH AT TIME OF SYSTEM EXEC.
#
#    * Same issue for learning in Process??
#
#    * Also, which execution_token should be used for simulations (while simulation uses actual system rather than copy)
#
# FIX:
# Finish Run:
#     assignment of inputs (for both Process and System):  consolidation from process and system execute methods
#
# Rename INPUTS -> STIMULI
# FIX: process.run crashes if stimuli are in dict format and there is more than one execution set
#                 (see Multilayer Learning Script)
#
# FIX: MAKE SURE SAME ORIGIN FOR DIFFERENT PROCESSES IS NOT ASSIGNED DIFFERENT PHASES
#
# FIX:
#
# System:
#    Finish implementing SystemStimulusInputs

# ObjectiveMechanism:
#    Add matrix assignments (and allow None to suppress it)
#    Add input assignments

# LearningProjection:
#    Finish implementing Comparator version of ObjectiveMechanmism

# FIX: EVC Gratton Script_Bug5:  Can't assign Linear() directly to intensity_cost_function (had to assign .function)
# FIX: When running a process with a TERMINAL mechanism that is also in another process, it gets input from that
#        process even if it is not running
# FIX: Can't specify parameter as ControlProjection (StroopEVCforDST)

# DOCUMENTATION COMPLETION/CLEAN-UP:
#   Function
# √ System
# √ Process
# √ Mechanism
#   ProcessingMechanism
#   DefaultProcessingMechanism
# √ DDM
# √ IntegratorMechanism
# √ TransferMechanism
# √ MonitoringMechanism
#   DefaultMonitoringMechanism
# √ ComparatorMechanism
# √ WeightedErrorMechanism
# √ ControlMechanism
#   DefaultControlMechanism
# √ EVCMechanism
# √ ControlSignal
# √ Projection
# √ MappingProjection
# √ ControlProjection
# √ LearningProjection
# √ State
# √ InputState
# √ ParameterState
# √ OutputState
# √ Run
#   Preferences
#   Log
#   TimeScale
#   Registry
#
# FIX: MAKE SURE SAME ORIGIN FOR DIFFERENT PROCESSES IS NOT ASSIGNED DIFFERENT PHASES

# DOCUMENT: targets argunment in system() and System_Base.__init__()

# DOCUMENT: ADD CHAIN EXAMPLE TO System AND Mechanism DOCSTRINGS
#
# DOCUMENT: FINISH DOCUMENTING:
#             .. _ControlMechanism_Specifying_Control:
#
#             Specifying control for a parameter
#             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DOCUMENT: it`s -> its (unless contraction)
# DOCUMENT:  ControlSignal/ControlProjection:
#                      not just mechanism or its function, but also a mapping projection;  reword referent as ``owner``
#            `ORIGIN` -> 'ORIGIN'
# DOCUMENT: MONITOR_FOR_CONTROL -> EVALUATE_FOR_CONTROL
# DOCUMENT:  MonitoredOutputStates -> EvaluatedOutputStates

# DOCUMENT:  Component:  under assign_params, document that parameter must be reference using a string that is the name
#                        of the argunent used for the parameter in the component's constructor (or the corresponding
#                        keyword, which is a capitlizaed version of its name, including any underscore separators )
#                        GIVE EXAMPLES.

# DOCUMENT: EVCMechanism NOTES ON API FOR CUSTOM VERSIONS:


# DOCUMENT:  UserDefinedFunction API:  wraps custom function, that can then be called using its function method;
#                can take variable, params, time_scale, and context as params, along with any of its own

#           FROM EVCMechanism.control_signal_grid_search:
#             Gets controller as argument (along with any standard params specified in call)
#             Must include **kwargs to receive standard args (variable, params, time_scale, and context)
#             Must return an allocation policy compatible with controller.allocation_policy:
#                 2d np.array with one 1d array for each allocation value
#
#             Following attributes are available:
#             controller.run: executes a specified number of trials with the simulation inputs
#             controller.predicted_inputs: ndarray of current value of outputState
#                                          for each predictionMechanism in self.system.prediction_mechanisms
#             controller.monitored_states: list of the mechanism outputStates being monitored for outcomes
#             controller.inputValue: list of current outcome values for monitored_states
#             controller.controlSignals: list of controlSignal objects
#             controlSignal.allocation_samples: set of samples specified for that controlSignal
#             [TBI:] controlSignal.allocation_range: range that the controlSignal value can take
#             controller.allocation_policy: current allocation_policy
#             controller.outputValue: list of current controlSignal values
#             controller.value_function: calls the three following functions (done explicitly, so each can be specified)
#             controller.outcome_aggregation function: aggregates outcomes (using specified weights and exponentiation)
#             controller.cost_function:  aggregate costs of control signals
#             controller.combine_outcome_and_cost_function: combines outcomes and costs



# DOCUMENT: Learning and Control in System
# DOCUMENTATION: go through DDM;  update refs, and add attributes for params (drift rate, starting_point, etc.)

# DOCUMENT: TO IMPOSE A DEFAULT PARAMETER CONDITIONALLY, NEED TO OVERRIDE _instantiate_parameter_states
#            (EXAMPLE:  TransferMechanism:  impose default range if function is Logistic)

# DOCUMENTATION: replace ``variable`` with
#                    :py:data:`variable <Module.variable>` or
#                    :py:data:`variable <Component.variable>`
#                same for ``function`` and ``value``

# DOCUMENTATION: Update LearningProjection_TERMINAL_vs_TARGET_fig
# DOCUMENT: LearningProjection.monitoringMechanism attribute
# DOCUMENT: Targets are checked against range of source mechanism (by Comaprator)
#           Full set of targets and inputs are generated by run??
# *******************************************************************************************************************
#
# DOCUMENTATION: params dictionary -> ``params`` dictionary
#                parameter dictionary -> ``params`` dictionary
#               System -> Agent?
#               Mechanism -> Process? [Representation? Transformation?]
#               phase -> event
#               MappingProjection matrix -> weightMatrix;  make corresponding changes in learningSignal

# DOCUMENT: In Components, document use of params dictionaries and/or assign_params methods for modifying
#                the parameters of a component "permanently";  describe relatinoshipo of keywords for parameters
#                which are simply convenience string constants that are the same as the name of the argument
#                for the parameter in the component's constructor. (see :ref:`EVCMechanism_Creation` for text)

# DOCUMENT: inputValue and outputValue are lists for convenience of user access, whereas
#           variable and value are 2d np.arrays that are used as internal datastructures
#


# DOCUMENT:  Explain better the relationship of an inputStates variable to its value, and of thes to the
#            to the potential for multiple items of a mechanism's variable (with an example:  ComparatorMechanism)


# DOCUMENT:  FIGURES FOR:
#                   inputState: inputValue vs. variable
#                   OutputState: function vs. calculate, value vs. outputState.value vs. outputValue

# DOCUMENT:  IN MECHANISM, CLARIFY:
#                DISTINCTION BETWEEN THE RESULT OF ITS FUNCTION,
#                                     AND ITS ``value`` ATTRIBUTE, WHICH IS WHAT IS RETURNED BY ITS EXECUTE METHOD
#                DISTINCTION BETWEEN ``value``, WHICH IS THE RESULT OF THE MECHANISM'S EXECUTION,
#                                AND ``outputValue``, WHICH IS A SUMMARY OF THE ``value`` OF EACH OF ITS outputStates
#
# DOCUMENTATION: add show to Systsem and Process
#

# DOCUMENT:
#    - Clean up documentation at top of module
#    - change relevant references to "function" to "execute method"
#    - note that run time params must be in FUNCTION_PARAMS
#    - Add the following somewhere (if it is not already there):
#     Parameters:
#         + Parameters can be assigned and/or changed individually or in sets, by:
#             including them in the initialization call (see above)
#             calling the adjust method (which changes the default values for that instance)
#             including them in a call the execute method (which changes their values for just for that call)
#         + Parameters must be specified in a params dictionary:
#             the key for each entry should be the name of the parameter (used to name its state and projections)
#             the value for each entry can be a:
#                 + State object: it will be validated
#                 + State subclass: a default state will be implemented using a default value
#                 + dict with specifications for a State object to create:
#                 + ParamValueProjection tuple: a state will be implemented using the value and assigned the projection
#                 + projection object or class: a default state will be implemented and assigned the projection
#                 + value: a default state will be implemented using the value

# DOCUMENT:
#    README.md -> README.rst AND/OR Index.rst:
#    Features:  graph support, logging, hierarchical preferences
#
#    System:  Control and Learning under Structure
#    OutputStates: INDEX argument, customizability, balance between customized outputStates and dedicated mechanisms

# DOCUMENT: Component:  :keyword:`NotImplemented` can be assigned to a parameter in the definition of paramClassDefaults
#                          to allow it to pass _validate_params without having to make an assignment (i.e., to
#                          suppress type checking.
#                       `None` is used to suppress its use by higher level assignments

# DOCUMENT UNDER ParameterStates
#     If parameter default value is set to None (or a non-numeric value),
#           either in paramClassDefaults, as default in constructor argument, or specified as such,
#           then no parameter state is created and can't be used either for Control, Learning or runtime assignment
#     Instantiate parameterState for each param in owner.user_params
#     - including ones in owner.user_params[FUNCTION_PARAMS]
#     - exclude if it is:
#        assigned a non-numeric value (including None, NotImplemented, False or True)
#           unless it is:
#               a tuple (could be on specifying ControlProjection, LearningProjection or ModulationOperation)
#               a dict with the name FUNCTION_PARAMS (otherwise exclude)
#        a function
#            IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
#            (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
#            i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)

# IMPLEMENT / DOCUMENT:
#             IMPLEMENTATION NOTE:  Process._execute_learning - ~line 1909
#                This implementation restricts learning to parameterStates of projections to inputStates
#                That means that other parameters (e.g. object or function parameters) are not currenlty learnable
#             ADD LEARNING TO OF OTHER PARAMETER STATES (E.G., OBJECT ITSELF AND/OR ITS FUNCTION)

# DOCUMENTATION: direct call to run or execute for mechanism executes its function in isolation
#                 (i.e., does not do any state updating), so can't use run_time params
#
# DOCUMENTATION:  runtime_param specification can use tuple, which specifies modulation operation for runtime param
#                       (including override)
# DOCUMENTATION: runtime_param can be specified for the parameters of an object or of its function,
#                but *NOT* the function itself  (COMMENT: problem is the possible need to re-instantiate the function;
#                paramInstanceDefaults may have just has the class, not an instance; could rectify by assigning that to
#                paramInstanceDefaults (see 11/27/16 in Component.instantiate_function)
# DOCUMENTATION: for now, only numeric parameters can be subject to control (may change in the future)
#                 COMMENT:  will need to:
#                                      allow all params to be assigned in Mechanism.execute (line 1336-7):
#                                             for param in self.function_params:
#                                                 runtime_params[param] = self.parameterStates[param].value
#                                      return keyword values (cf. commented out keyword method for Function.Integrator
#                                      handle management of contraints in State._instantiate_state (line 1869):
#                                               constraint_value = convert_to_np_array(constraint_value,1)
#
# DOCUMENTATION:  TAKE CARE OF THE FOLLOWING:
#         COMMENT:
#           MOVE THE BULK OF THIS TO THE DESCRIPTION OF RUNTIME PARAMS ABOVE, AND REFERENCE THAT.
#         COMMENT

# DOCUMENTATION:  singularize first statement in overview of all objects
#
# DOCUMENTATION:  SEARCH FOR :class: AND REPLACE WITH :py:class:

# DOCUMENTATION: check that, for DDM in TIME_STEP mode, parameter values not specified in params dict will assume
#                any value assigned in the function arg; otherwise, default will be used
# DOCUMENTATION:  FROM DDM:  PUT FULL EXPLANATION IN MECHANISM AND THEN ADD LINK TO BELOW
#             since any parameters specified in the  ``params`` argument when
#             creating a mechanism override any corresponding ones specified as arguments to its ``function``).[LINK]

# DOCUMENTATION: MONITOR_FOR_LEARNING (in LearningProjection AND ??WHERE ELSE:
#                                                                 Mechanism?? Paralleling MONITOR_FOR_CONTROL
#                                                                 OutputState??
#                                                                 MonitoringMechanism??
# DOCUMENTATION: THESE NEED TO BE ADDED:
#       (see :ref:`ControlMechanisms_Monitored_OutputStates` for details of specification).
#       (see :ref:`MonitoringMechanisms_Monitored_For_Learning` for details of specification).

# DOCUMENTATION: MOVE DESCRIPTION OF PARAMETER SPECIFICATION DICTIONARY FROM UNDER MECHANISM TO UNDER COMPONENT
#                  AND ADJUST ALL REFERENCES OF THE FOLLOWING TYPE ACCORDINGLY:
#                   (see :doc:`Mechanism` for specification of a parms dict)
# DOCUMENTATION:  NEED GENERAL INTRO, INCLUDING COMMENT ABOUT SPECIFYING ARGUMENTS/PARAMETERS:
#                    FOR ARGUMENTS OF __init__ , THERE IS USUALLY AN ATTRIBUTE OF THE OBJECT THAT CAN BE ASSIGNED A
#                    VALUE AFTER IT IS CREATED.  (PUT THIS WHEREVER PARAMS, PARAMSCURRENT, INSTANCE DEFAULTS ETC.
#                    ARE DISCUSSED.
# DOCUMENTATION: ControlMechanism -> controlMechanism or control mechanism (in appropriate places)
# DOCUMENTATION: Call subclass -> "Constructor"

#  DOCUMENTATION: Learning -> LearningProjection (name of doc)
#
#  DOCUMENTATION: add the following to attributes of class:
#                object-specific params to list of
#                function_params
#                consider adding the following (paralleling Projection):
#                     params : Dict[param arg, parm value]
#                         set currently in effect
#
#                     paramsCurrent : Dict[param arg, parm value]
#                         current value of all params for instance
#
#                     paramInstanceDefaults : Dict[param arg, parm value]
#                         defaults for instance (created and validated in Components init)
# DOCUMENT:
# If the sender outputState and/or the receiver inputState are not specified:
#    - a mapping will be created for only sender.outputState and receiver inputState (i.e., first state of each)
#    - the length of value for these states must match
#
# - DOCUMENT: Finish editing Description:
#             UPDATE TO INCLUDE Mechanism, Projection, Mechanism FORMAT, AND (Mechanism, Cycle) TUPLE
#
# DOCUMENT:
#     SHOULD ALSO BE INCLUDED IN DOCUMENTATION OF EXECUTE METHOD FOR PROCESS AND SYSTEM:
#     *Number of phases (time_steps) per trial.* Processes have only one phase per trial, but systems can have
#     more than one.  If the mechanisms in a system use more than a single phase, then the next level of
#     nesting of a lists, or next higher axis of an ndarrays is used for the sequence of phases.

# DOCUMENT: update System docstring for MechanismList and mech_tuple attributes (following format of Process)

# DOCUMENT parsing of (paramValue, projection_spec) tuples:

# DOCUMENT:
# ORDER OF INSTANTIATION OF PARAMETER STATES AND EXECUTE METHODS
# ORDER OF INSTANTIATION OF LEARNING PROJECTION COMPONENTS:
# _deferred_init FOR LEARNING PROJECTIONS, MAPPING_PROJECTIONS W/O RECEIEVERS, ETC.

# DOCUMENT:
    #     ComparatorMechanisms ARE INCLUDED FOR EXECUTION DURING LEARNING,
    #     BUT NOT FOR REPORTING, AND SHOULD NOT BE CONSIDERED TERMINALS FOR EVC MONITORING
    #      FOR OPTIONAL INPUT, AND ADD ARGUMENT TO SYSTEM FOR ASSIGNING INPUT AT EXECUTE TIME
#endregion
#
# DOCUMENT:  Add links throughout for time_step and trial

#region DEVELOPMENT
# time_step DDM integration
#
# system.graph -> NetworkX, graphViz, or the like.
#
# API for "wrappers"
#
# IMPLEMENT:  Demos of Functions that plots each Function
#                                (use new "demoRange" attribute that specifies range of inputs for Function for demo)
#
# GUI (using QT)
#
# FIGURE OUT HOW TO GET DILL WORKING TO CACHE SYSTEM IN System._cache_state, OR STORE AS BINARY OBJECT
#
# function format for inputs in Run (e.g., for simulating staircasing):
#                       needs to be coordinated with validation of num_targets (if it is not a function);
#                       then document (paralleling targets, and maybe moving much of that to inputs function format)
#
# Filter Warnings
#
# TEST run(inputs) dict format FOR SITUATION IN WHICH TWO PROCESSES HAVE THE SAME INPUT,
#                                                     OR ONE PROCESS BRANCHES OUT FROM ANOTHER
#                  ISSUE IS WHETHER THE RIGHT NUMBER OF INPUTS ARE ASSIGNED, AND WHETHER SYSTEM KNOWS NOT TO
#                  CREATE AN INPUT FOR THE BRANCHING PROCESS (SEE RUN line 688 and SYSTEM line 1388
#
# README.md -> README.rst AND/OR Index.rst:
#endregion

#
#region PY QUESTIONS: --------------------------------------------------------------------------------------------------

# QUESTION:  how to initialize a numpy array with a null value, and then assign in for loop: np.empty
#endregion
# -------------------------------------------------------------------------------------------------

#region PYCHARM QUESTIONS: ---------------------------------------------------------------------------------------------

# QUESTION:  how to identify method in which breakpoint has occurred (or where execution has paused/stopped)
# QUESTION:  how to share breakpoints across installations?
# QUESTION:  how to set default for expanded vs. collapsed Favorites window/pane

#endregion
# -------------------------------------------------------------------------------------------------

#region DEPENDENCIES: -----------------------------------------------------------------------------------------------
#
# toposort.py
# wfpt.py
# mpi4py.py
#
#region BRYN: -------------------------------------------------------------------------------------------------------

# Documentation (from code -> HTML/MD -> website
# Jupyter and matplotlib

# - QUESTION: Better way to do this (check for a number or 0D np value and convert to 1D?):
#             if isinstance(target, numbers.Number) or (isinstance(target, ndarray) and target.ndim == 0):
#                 target = [COMPARATOR_TARGET]
#             # If input is a simple list of numbers (corresponding to 0D), wrap in an outer list (i.e., make 1D)
#             if all(isinstance(i, numbers.Number) for i in target):
#                 target = [COMPARATOR_TARGET]
# - QUESTION: OK to have mutable objects in arguments to init?? (e.g., System)
# - QUESTION:
#   How to avoid implementing DefaultController (for ControlSignals) and DefaultTrainingMechanism (for LearningSignals)
#   and then overriding them later??

# - ABC
# - params dict vs. args vs. **kwargs:  FIX: LOOK AT BRYN'S CHANGES TO isCompatible
# - FIX: LOOK AT HIS IMPLEMENTATION OF SETTER FOR @ClassProperty
# - QUESTION: CAN ERRORS IN TypeVar CHECKING BE CAPTURED AND CUSTOMIZED?
#            (TO PROVIDE MORE INFO THAN JUST THE ERROR AND WHERE IT OCCURRED (E.G., OTHER OBJECTS INVOLVED)
# - Revert all files to prior commit in PyCharm (VCS/Git/Revert command?)


# It’s helpful if methods that mutate object state have names that suggest they will do so.
#      For example, it was confusing to me that _validate_variable assigns self.variable and self.variableClassDefault
#      (at least it does in Mechanism, I’m not sure about other subclasses).  I was expecting it simply to validate,
#      as in do nothing if the variable was OK, and throw an exception if it wasn’t.
#      It may sound kooky, but even a clunky name like “validate_and_set_variable” would be better,
#      or better still would be to make validate really just validate,
#      and have another method like “assign_variable” or something.
# In general, every assignment statement changes the behavior of the program in ways that are non-local,
#      and thus hard to understand and debug. So you want as few of them as you can possibly get away with,
#      and you want them clearly identified.
# NotImplemented is used a lot for missing arguments. Usually people use None for that.
#     This also allows for a nice idiom. Given a parameter foo = None, you can do defaulting like this:
#     myval = foo or “some default value”
#     So you get myval = foo if foo is truthy (not None, [], (), 0, 0.0, “”, or False), and your default value otherwise
# I don’t think you have to worry quite so much about people implementing classes wrongly or subversively.
#     This is Python - if they want to do bad things, you’ll be hard pressed to stop them.
#     All you can do is guide them in the right direction.
# In Function there’s a test to make sure there’s a registry - this probably ought to be handled by having
#     a base class “Category” or something, that ensures there is one in its __init__, and just insisting that
#     every category class extends that “Category” class. We can talk more about this.
# Normally when implementing __init__, it’s a good idea for base classes to call super().__init__ first,
#     before doing anything else. In some languages, e.g. C++ and Java, it’s actually required to be the first thing.
#     There were some comments in Function.__init__ that made me think you’re expecting people to do some setup before
#     calling super().__init__.
# Automated type checking (like typecheck-decorator) would reduce code size noticeably.
# PEP8
#     Rename packages lowercase, Components -> functions
#     Mechanism_Base -> MechanismBase
#     Method names, e.g. verbosePref -> verbose_pref in ComponentPreferenceSet
#     Aim for methods that fit on a single screen, e.g. Function.__init__ is about 150 lines,
#         but you want something like 50 lines for a method for it to be comprehensible.
#     Along the same lines, too many #regions, and too much SHOUTING. Breaks up the reader’s flow.
#     Single line comments in normal case are fine.
#     No need for #regions around 1-2 lines of code, especially if your region name almost exactly matches
#         the name of the method you’re calling in the region (e.g. Function, around the end of __init__)
#     For each #region more than 2-3 lines long, consider whether it would be better to extract that code to
#         a small helper method or function.
# Commenting style:
#     Want comments on each method, not one block at the class level that lists all the methods.
#     Documentation generators like sphinx will generate those class summaries from component parts,
#         no need to synthesize them yourself
#     Guiding principle: docs as physically close to the code as possible, so less likely to get out of sync.
#     No point in listing things like “:param level: and :return:” if they’re not actually going to be documented,
#         it’s just taking up space.
#     Lots of code commented out. Just delete it, git will get it back for you if you decide you need it again.
#     Use doc strings to document class members, not comments
#         (e.g. Function.py line ~207 doc for variableClassDefault_Locked)

#endregion

#region PNL JAMBOREE
#
# DEVELOPMENT:
# TimeStep time scale (DDM & TransferMechanism Mechanisms)
# Implement single centralized registry
# Learning execution sequence
# Cyclic system
# API / extensibility
#
# ACTION ITEMS:
#
# Flatten params (and add kwArgs handling) to Functions
# - function -> function
# - functionParams -> args and/or params dict inside Function specification
# - if functionParams are now all handled inside specification of a Function for function param:
#      - need to make sure all parses of function can now handle this
#      - instantiation of parameterStates needs to extract any params specified as args and/or in a params dict
# - add @property for all params, so they can be addressed directly as attributes
#      setter method should call _assign_defaults
#

#endregion

#region EVC MEETING: ---------------------------------------------------------------------------------------------------

# -------------------
#
# QUESTION:     Should ControlSignal "intelligence" (e.g., intensity, costs, etc.)
#                   be in EVC mechanism rather than ControlProjection?
#                   EVCMechanism makes more sense theoretically
#                      e.g., seems better to talk about the cost of a control signal, rather than ControlProjection
#                   Projection is easier to manage (as it is its own object, can be parameterized, etc.)
#                   Maybe they should be assigned to OutputStates of the EVCMechanism?
#
# QUESTION: DDM:
#            MULTIPLE PROCESSES AND AVERAGED OUTPUT VALUES IDEA
#            t0 BUISINESS (ms or secs?)
#            DOES NAVARRO AND FUSS ACTUALLY RETURN ER (I ASSUMED IT DID)
#            DOES NAVARRO AND FUSS ACTUALLY RETURN MEAN RT FOR CORRECT RESPONSE?  SHOULD THIS TOO BE UPPER BOUND??
#            HOW TO DESCRIBE RESULTS OF INTERROGATOIN PROTOCOL (IN TIME_STEP MODE)
#            IS t0 MS OR SECONDS?

# -------------------------------------------

# FIX: PROCESS INPUT, AND TARGET INPUT TO ComparatorMechanism, ARE RESTRICTED TO PROCESS TO WHICH MECHANISM BELONGS
#      ?SHOULD SAME BE TRUE FOR ALL PROJECTIONS:  ONLY UPDATE THOSE BELONGING TO MECHANISMS WITHIN THE PROCESS?

# QUESTION: When executing a mechanism, and updating its projections (to get their input),
#               should update only those from mechanisms that belong to the process currently being executed?
#               or should all projections be updated (irrespective of source) when executing a mechanism?
#           This issue includes Process inputs, as well as target inputs
#           Inclined to only restrict Process and target inputs
#           (since those are process-specific) but not other projections
#
# QUESTION: RL:
#       Option 1 - Provide Process with reward for option selected: more natural, but introduces timing problems:
#           - how to provide reward for outcome of first trial, if it is selected probabilistically
#           - must process trial, get reward from environment, then execute learning
#           SOLUTION: use lambda function to assign reward to outputState of terminal mechanism
#       Option 2 - Provide Process with reward vector, and let targetMechanism choose reward based on action vector
#           - softamx should pass vector with one non-zero element, that is the one rewarded by comoparator
#           SOLUTION:  use this for Process, and implement Option 1 at System level (which can control timing):
#           - system should be take functions that specify values to use as inputs based on outputs
#               as per SOLUTION to Option 1 using lambda functions

# QUESTION: Default object (e.g., default_projection for Process): should they be templates or objects?
#                                                                  or signify (e.g., class = template)
#
# QUESTION: ??OPTION (reshapedWeightMatrixOption for MappingProjection) TO SUPPRESS RESHAPING (FOR FULL CONNECTIVITY)
#
# QUESTION: WHICH CLASS SHOULD HANDLE THE EXECUTION OF LEARNING:  PROCESS OR SYSTEM
#           Process:
#               - it manages the instantiation of LearningSignals
#               - part of the definition of a Process: output is where supervised training signals are provided
#               - may want to build a model that can learn and only has a Process
#           System:
#               - need to manage execution of learning in systems anyhow (same as for mechanisms)
#               - learning that needs to straddle Processes
#                   (e.g., error-signals that need to be passed from the first layer of one Process
#                    to the last layer of a preceding Process) - but then make them one Process (per definition above)?
#
# FIX: HOW IS THIS DIFFERENT THAN LENGTH OF self.variable
#         + kwTransfer_NUnits (float): (default: Transfer_DEFAULT_NUNITS
#             specifies number of units (length of input array)
#
# NEED FOR EVC MODEL:
# - Sequential adjust effects:
#   "Reactive":  simple controlMechanism that maps input values into ControlProjection intensities
#   "Simple Exhaustive Search": find optimal policy for stimulus/reward values
#   "Feature-based model learning" (Falk & Tom)
#   "Exhaustive Search + learning":
#       searches through all ControlSignals to find the optimal one
#       stimulus prediction
#       reward prediction
#       automatic component of the drift for each stimulus (== weight matrix)
#    *  d(parameter_value)/d(control signal intensity) for each control signal ==
#                                                          differential of the parameterModulationFunction
#       NOTE:
#          THIS IS DISTINCT FROM THE ControlProjection.function
#                                           (== intensity_function) WHICH MAPS ALLCATION -> ControlProjection Intensity
#          BUT IS ISOMORPHIC IF ControlProjection.function IS Linear with slope = 1 and offsent 0 (i.e,. its default)
#       QUESTION:  DO WE CARE ABOUT THE DIFFERENTIAL ON ALLOCATION -> parameter_value (.e., ControlSiganl.function)
#                       OR ControlProjection Intensity -> parameter_value (i.e., parameterModulation function)??
#        SEBASTIAN FAVORS LEAVING IT AS DIFFERENTIAL ON parameterModulation function
#    *  Parameters of parameterModulation function should be accessible

#endregion

#region GENERAL: -------------------------------------------------------------------------------------------------------
#
# - Register name:
#    PsyNeuLink
#    [PsyPy? PsyPyScope?  PyPsyScope?  PsyScopePy? NeuroPsyPy?  NeuroPsySpy]
#
# Search & Replace:
#   show() -> show()
#   ControlProjection -> ControlProjection
#   LearningSignal -> LearningProjection
#   "execute method" -> function:  BUT NEED TO BE CAREFUL, SINCE "<object>.execute method" SHOULD *NOT* BE REPLACED
#   <>.paramsCurrent = <>.params
#   kwXxxYyy -> XXX_YYY
#   MATRIX -> kwWeightMatrix;  matrix -> weightMatrix in MappingProjection
#   item -> element for any array/vector/matrix contexts
#   function (and execute Method) -> executeFunction (since it can be standalone (e.g., provided as param)
#   PARAMETER_STATE -> PARAMETER_STATES
#   MechanismParamValueparamModulationOperation -> MechanismParamValueParamModulationOperation
#   functionParams -> ParameterStates
#   InputStateParams, OutputStateParams and ParameterStateParams => <*>Specs
#   KwDDM_StartingPoint -> DDM_StartingPoint
#   CHANGE ALL VARIABLES FROM THEIR LOCAL NAMES (E.G., Allocation_Source, Input_value, etc) to variable
#   Projections: sendsTo and sendsFrom
#   "or isinstance(" -> use tuple
#   Change "baseValue" -> "instanceValue" for prefs
#   super(<class name>, self) -> super() [CHECK FUNCTIONALITY IN EACH CASE]
#   NotImplemented -> None (and adjust tests accordingly)
#
# FIX: execute VS. update
#      SUTBTYPES DON'T CURRENTLY IMPLEMENT update();  THEY USE execute() for both housekeeping and function
#      WOULD BE BETTER AND MORE CONSISTENT TO HAVE THEM IMPLEMENT update() WHICH CALLS .execute
#      PROBLEM with implementing update() in subclasses of Mechanism:
#          1) it would override Mechanism.update so must call super().update
#          2) no obvious place to do so, since execute() (which MUST be implemented by subclass)
#                 is called in the MIDDLE of Mechanism.update
#          3) could name subclass.update() something else, but that gets more complicated
#      PROBLEM with NOT implementing update() in subclasses of Mechanism:
#          1) they are now special (i.e., not treated like all other classes in Function hierarchy)
#          2) all of the "housekeeping" for execution must be done in subclass.execute()
#          3) that means that kwExecute can't be used to override self.execute (i.e., defeats plug and play)
#      CURRENT SOLUTION:
#          use FUNCTION as scripting interface
#          intercept specification of FUNCTION before _instantiate_function (e.g., in _validate_params),
#              reassign to instance attribute, and del FUNCTION from paramsCurrent
#
# - FIX: get rid of type/class passing
#        - replace all type/class specifications of params with kw string specifications for each type
#        - implement global and/ or local lookup table(s) of types (locally for each (set of) types)
#        - reserve a special keyword (e.g, CLASS_NAME or kwClassName or kwDEFAULT) for specifying default of instance
#
# FIX: NotImplemented STILL LINGERING FOR:
#                MONITOR_FOR_CONTROL (see Mechanism line 696)
#                     needed to distinguish between suppress and ignore MONITOR_FOR_CONTROL
#                variableClassDefault (see Component line 408)
#                    needed to allow Process and System to set their variableClassDefault values to None
#
# - FIX: GET RID OFF '-1' SUFFIX FOR CUSTOM NAMES (ONLY ADD SUFFIX FOR TWO OR MORE OF SAME NAME, OR FOR DEFAULT NAMES)
# - FIX: MAKE ORDER CONSISTENT OF params AND time_scale ARGS OF update() and execute()
#
# TEST: RUN TIMING TESTS FOR paramValidationPref TURNED OFF
#
# TEST warnings.warn
#
# IMPLEMENT:  typecheck name arg in constructors to be a str
#
# IMPLEMENT: REFACTOR EVC and LEARNING:
#
#            EVC:  1) MonitoringMechanism - new one that implements current EVCMechanism's objective function (i.e.,
#                                            (i.e., using LinearCombination).
#                  2) EVCMechanism - new version that takes output of MonitoringMechanism, and just handles search.
#                  3) ControlProjection - as it is now.
#
#            Learning: 1) MonitoringMechanism - use relevant one, as it does now.
#                      2) LearningMechanism - move BP and RL implementations from LearningProjection to this/these;
#                                               takes output of MonitoringMechanism (as errorSignal)
#                                               as well as an errorSource as its input
#                                               (rather than figuring it out, as it does now);
#                                               BP version calculates gradient on that;  RL does its thing;
#                                               generates modification specification for LearningProjection.
#                      3) LearningProjection - simplified version (paralleling ControlProjection implementation):
#                                                * takes modification specification from LearningMechanism and
#                                                     formats it for item being modified (learned);
#                                                * can be used for MappingProjection or ProcessingMechanism.
#
#
# - IMPLEMENT: Config (that locally stashes default values for user)
#
# - IMPLEMENT: integrate logging and verbose using BrainIAK model:
#              no printing allowed in extensions
#              verbose statements are logged
#              log goes to screen by default
#              can define file to which log will go
#
# - IMPLEMENT: master registry of all Function objects
# - IMPLEMENT TYPE REGISTRIES (IN ADDITION TO CATEGORY REGISTRIES)
#
# - IMPLEMENT switch in __init__.py to suppress processing for scratch pad, etc.
#
# IMPLEMENT: Quote names of objects in report output
#
# IMPLEMENT: Change all enum values to keywords (make read_only?? by using @getters and setters)
#            (follow design pattern in SoftMax)
#
# - IMPLEMENT Testing:
#     use instantiation sequence (in Utilities) to create test for each step
#
# - Fully implement logging
#    For both of the above:
#       use @property to determine whether current value should be set to local value, type, category or class default
# - Implement timing
# - implement **args (per State init)
# - MAKE SURE _check_args IS CALLED IN execute
#
# - iscompatible:
# -   # MAKE SURE / i IN iscompatible THAT IF THE REFERENCE HAS ONLY NUMBERS, THEN numbers_only SHOULD BE SET
# -   Deal with int vs. float business in iscompatible (and Function_Base functionOutputTypeConversion)
# -   Fix: Allow it to allow numbers and strings (as well as lists) by default
#     and then relax constraint to be numeric for InputState, OutputState and ParameterState
#     in Mechanism._validate_params
# -   Implement: #  IMPLEMENTATION NOTE:  modified to allow numeric type mismatches; should be added as option in future
#
# IMPLEMENT: add params as args in calls to __init__() for Function objects (as alternative to using params[])
#
# MAKE CONSISTENT:  variable, value, and input
#
# - Registry:
#   why is LinearCombination Function Components registering an instanceCount of 12 but only 2 entries?
#   why is DDM registering as subclass w/o any instances?
#   why are SLOPE and INTERCEPT in same registry as Statess and Parameters?
#   IMPLEMENT: Registry class, and make <*>Registry dicts instances of it, and include prefs attribute
#
# IMPLEMENT: change context to Context namedtuple (declared in Globals.Keywords or Utilities):  (str, object)
#
#endregion

#region SCRIPTS: -----------------------------------------------------------------------------------------------

# IMPLEMENT: Extend Multilayer Learning Test script to use multiple forms of parameter specification

#endregion

# region DEPENDENCIES:
#   - toposort
#   - mpi4py
#   - wfpt.py
# endregion

# region OPTIMIZATION:
#   - get rid of tests for PROGRAM ERROR
# endregion

#region DOCUMENTATION: -------------------------------------------------------------------------------------------------

# QUESTION: should attributes that are common to different subclasses be documented there, or only in the base classes?

# DOC LIST:

#    IntegratorMechanism
#  √ ComparatorMechanism
#  ~ ControlMechanism
#  √ ControlProjection
#  ! DDM
#    DefaultControlMechanism
#  √ EVCMechanism
#    Function
#  ~ InputState
#  √ LearningProjection
#    Log
#  √ MappingProjection
#  ! Mechanism
#  √ MonitoringMechanism
#  ~ OutputState
#  ~ ParameterState - NEED TO ADD RUNTIME SPECIFICATION
#    Preferences
#  √ Process
#    ProcessingMechanism
#  ~ Projection
#  ! Run
#  √ State
#  ! System
#  ! TransferMechanism
#    Utilities
#  √ WeightedErrorMechanism

# SPHINX / RST ***********************************************************

# Convention for names of arguments, attributes, methods and keywords:
# =====================================================================

# argument_attribute -> argument and user-accessible attribute derived from a constructor argument
# nonArgumentAttribue -> user-accesible attribute that is not an argument in the constructor
# _internal_atttribute or _method -> not user accessible, and not to be included in rst docs
# KEY_OR_KEYWORD -> name of a str used as a key for a dict or as a PsyNeuLink keyword

# Main documentation of params/attributes should be in module docstring;  Arguments and Attributes should refer to that


# rST formatting for tokens:
# =========================

# None:
#    `None`

# Keywords:
#   :keyword:`<KEYWORD>`

# Arguments, parameters, attributes, methods and functions:
#   ``argument``, ``parameter``, ``attribute``, ``method``, or ``function``

# Section reference:
#    _<DOCUMENT>_<SECTION>_<SUBSECTION>:


# rST formatting for Headings:
# ===========================

# SECTION: -------
# SUB SECTION: ~~~~~~~
# SUB SUB SECTION: ..........
# EXCLUDE FROM DOCS: COMMENT:
#                    Text to be excluded
#                    COMMENT

# Arguments [SECTION]
# ---------

# <argument> : <type> : default <default>
#    description.  <- Note, first line of description is not capitalized (since it is prepended with hyphenation)
#    More description.

# Attributes [SECTION]
# ----------
# <attribute> : <type> : default <default>
#    Description. <- Note, first line of description IS capitalized (since it is NOT prepended with hyphenation)
#    More description.
#
#    .. _<attribute>  <- Internal attribute is commented out (by prepending dots and indentation)
#           Description.
#           More description.


# ?? ADD TO FUNCTION OR GENERAL DESCRIPTION SOMEWHERE:

    # Notes:
    # *  params can be set in the standard way for any Function subclass:
    #     - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
    #          and used for paramsCurrent unless and until the latter are changed in a function call
    #     - paramInstanceDefaults can be later modified using _assign_defaults
    #     - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

# FIX: Figures: need higher rez

# FIX:
# Where is this coming from:
#    Process.random() → x in the interval [0, 1).# Suppress / rearrange particular members:
#
# Suppress / rearrange particular members:
#    tc/typecheck decorators in argument lists
#    class types: e.g., namedtuples
#    specific definitions: e.g., ProcessRegistry
#    @property declarations (or group them with/as attributes?)
#    @<variable_name>.setter
#
# Line spacing between lines in a list

# Dereference variable values
#   Example (line 295 in DDM):
#      default_input_value : value, list or np.ndarray : :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`
# Any better way to format defaults in argument and attributes?  Is "default" a keyword for default or just a convention
# How to underline?
# Why does adding ": default _______ " to parameter specification suppress italicization??

# ADDITIONAL QUESTIONS / ISSUES:
# Why are some parameter type specifications (in parens) italicized and others not?
# Why do some underlines work and not others (e.g., Examples in Process)
# Definition of a Python keyword
# Why does Process_Base get referenced as Process, but System_Base as such?

# US:
#     Systematize #D vs. #d

# ***********************************************************************

# DOCUMENT: constructor_arguments get instantiated (if ncessary) and assigned to objectAttributes
#           members of params dicts get turned into object_attributes but remain in original state

# DOCUMENT: Lazy Evaluation:  PUT .. _Lazy_Evaluation: SOMEWHERE (IN COMPONENTS??)
# "lazy evaluation" (or call-by-need) (see https://en.wikipedia.org/wiki/Lazy_evaluation)[LINK].
# :ref:"Lazy_Evaluation": for execution, this means that objects are updated by
# calling items from which they receive input;  for implementation, this means that objects can create objects
# from which they expect input, but cannot "impose" the creation of "downstream" objects.

# DOCUMENT: Deferred Initialization
#   For LearingSignal, ??ControlProjection

# DOCUMENT: TARGETED FOR / INTENDED USES/USERS:
#                     OVERALL STRUCTURE, INCLUDING:  COMPONENTS MADE UP OF VARIABLE, FUNCTION AND OUTPUT
# DOCUMENT: TARGETED FOR / INTENDED USES/USERS:
#                novices (students, non-modelers)
#                "sketch pad", mock-up of models
#                integration of different components
#                model sharing/distribution, documentation, and archiving
#                small-moderate scale agential (e.g., social) interactions (1-10 participants)
#                not (yet?) optimized for:
#                           intensive model fitting, i.e.:
#                               generation of distributions of behavior
#                               automated parameter estimation
#                           large-scale simulations, e.g.:
#                              deep learning at the individual level
#                              population effects at the social level
#                           biophysics
#                           large-social interaction
#
#  CLEAN UP THE FOLLOWING
# - Combine "Parameters" section with "Initialization arguments" section in:
#              Function, MappingProjection, ControlProjection, and DDM documentation:

# DOCUMENT: SYSTEM:
#           ORIGIN: origin mechanism of a process in a system that does not receive projections from any other mechanisms
#                   NOTE: if a mechanism that is an origin for one process, but also appears as an INTERNAL mechanism
#                         in another process, it is NOT treated as an origin in the system;
#           INTERNAL: mechanism both receives projections from and sends projections to other mechanisms in the system
#           INITIALIZE_CYCLE: mechanism that has an outgoing projection that closes a cycle (feedback loop),
#                       so it should be properly initialized
#                       NOTE: self.executionGraph elides the projection that closes the loop so that an acyclic graph can be
#                             constructed to generate an execution list / sequence;  however, the projection is
#                             still operational in the system and will support recurrent (feedback) processing)
#           TERMINAL: terminal mechanism of a process that does not project to any other processing mechanisms
#                     (however, it can project to a ControlMechanism or a MonitoringMechanism)
#
# DOCUMENT: PROCESS:
#           If either the sender and/or receiver arg of a MappingProjection are not specified,
#               initialization of the projection is delayed.  This has the following consequence:
#           If the MappingProjection is defined outside the Process pathway and not explicitly listed in it,
#               it will not be included in the Process;  this is because deferring intialization means that
#               even if the sender or the receiver is specified, the projection will not be assigned to the
#               specified mechanism's projection list (sendsToProjections, receivesFromProjections), and thus not
#               identified in _instantiate_pathway.  Could allow sender to be left unspecified and still
#               proceed with initialization that thus be recognized by the Process;  however, can't do the reverse
#               (specify sender but not receiver) since receiver *must* be specified to initialize a projection
#               this assymetry might be confusing, and thus neither is allowed
#           However, if projection is listed in pathway, it is not necessary to specify its sender or receiver

# DOCUMENT: FUNCTION:
#           To use keywords for params, Function Function must implement .keyword method that resolves it to value
#           To use lambda functions for params, Function Function must implement .lambda method that resolves it to value

# DOCUMENT: MAPPING_PROJECTION:  different types of weight assignments
#            (in MappingProjection _instantiate_receiver and Function LinearCombination)
#            AUTO_ASSIGN_MATRIX: if square, use identity matrix, otherwise use full
#                                differs from full, in that it will use identity if square;  full always assigns all 1s

# DOCUMENT:  PROCESS: specifying the learning arg will add the LearningProjection specifcadtion to all default projections
#                      as well as any explicity specified (except for ones that already have a LearningProjection specified)

# DOCUMENT:  PROJECTIONS:  deferred init -> lazy instantiation:
#                          for MappingProjection and ControlProjection, if receiver is not specified in __init__,
#                              then iniit is deferred until State.instantiate_projection_to? from? is called on it
#                          for LearningProjection, at end of Process._instantiate_pathway
# DOCUMENT:  ARGS & PARAMS
# • Function:
#    CODE:
#    - assign_args_to_params makes specified args in __init__() available in <>.params (with keyword = arg's name)
#    SCRIPT:
#    - _assign_args_to_param_dicts() and _validate_params() now handle the following formats:
#                drift_rate=(2.0, CONTROL_PROJECTION),
#                drift_rate=(2.0, ControlProjection),
#                drift_rate=(2.0, ControlProjection()),
#                drift_rate=(2.0, ControlProjection(function=Linear)),
#                drift_rate=(2.0, ControlProjection(function=Linear(slope=2, intercept=10))),
#
# DOCUMENT: ASSIGNMENT OF DEFAULT PARAM VALUES:
#               For params not accessible to user:  assign params and default values in paramClassDefaults
#               For params accessible to user:  assign params and default values in args to __init__()
#               All subclasses of Function *must* include in their __init__():
# call _assign_args_to_param_dicts
#            PRINCIPLE:
#                 if there is ONLY one value for the function:
#                     - don't include as arg in __init__ (put in paramClassDefaults)
#                         (since there is no way to change it;  sacrifices power-coder's change to install their own)
#                     - include the function's arg as args in __init__
#                         (since there is only one set (no confusion of which belong to which of the possible functions)
#                          and it is more convenient than having to specify the function in order to specify its params)
#                     - package the args in function_args in _assign_args_to_param_dicts()
#                 if there is MORE than one value for the function:
#                     - include it as arg in __init__()
#                          (since there are different options)
#                     - do NOT include its args in __init__()
#                          (since some might be inappropriate for some functions)
#                     - they should be declared inside the definition of the function in the function arg
#
# DOCUMENT: Function Components don't use functionParams (i.e., they are the end of the recursive line)

# DOCUMENT: function, execute & update
#            .execute should be called to execute all object classes (except States):
#                it takes care of any "house-keeping" before and after it calls .function (if it exsits)
#                .execute should always return an array, the first item of which is the return value of .function
#                (note: System and Process don't implement a separate .function; it points to .execute)
#                Subclasses of mechanism implement _execute that is called by Mechanism
#                    - this is so Mechanism base class can do housekeeping before and after subclass._execute)
#                    - if a subclass does not implement _execute, calling it will call .function directly
#                    -  if INITIALIZING is in context for call to execute, initMethod is checked to determine whether:
#                        only subclass._execute is run (initMethod = INIT__EXECUTE__METHOD_ONLY)
#                        only subclass.function is run (initMethod = INIT_FUNCTION_METHOD_ONLY)
#                        full subclass._execute and Mechanism.execute method are run
#                States use .execute only to call .function (during init);  they are updated using <state>.update()
#            .function is the "business end" of the object:
#                - generally it is a Function Function
#                - but can be anything that adheres to the Function API

# DOCUMENT: Construction/Initialization Implementation:
# 1) Function implements _deferred_init(), which checks whether self.value is DEFERRED_INITIALIZATION;
#     if so, calls super(<subclass>,self).__init__(**self.init_args)
#     <subclass> is the class implementing deferred initialization
#     <**self.init_args> is the set of args passed to the __init__() method of the subclass
# 2) an object can defer initialization by doing the following in its __init__ method: (see LearningProjection for example)
#     - storing its args as follows:
#         self.init_args = locals().copy()
#         self.init_args['context'] = self
#         self.init_args['name'] = name
#         del self.init_args['self']
#     - set self.value = DEFERRED_INITIALIZATION
# 3) Where projections are ordinarily instantiated, assign instantiated stub" to sendsToProjections,
# 4) When a process is instantiated, the last thing it does is call _deferred_init
#    for all of the projections associated with the mechanism in its pathway,
#    beginning with the last and moving backward though the pathway
# 5) When finally instantiating deferred projections, be sure to do validation of their variable with sender's output:
#          State._instantiate_state:  elif iscompatible(self.variable, projection_spec.value):
# 6) update() method should test for self.value and if it is DEFERRED_INITIALIZATION it should return self.value
# 7) Objects that call execute method of ones with deferred init should test for return value of DEFERRED_INITIALIZATION
#     and handle appropriately


# DOCUMENT: LEARNING
#  Principles:
# - learning occurs on processes (i.e., it has no meaning for an isolated mechanism or projection)
# - Initialization of LearningSignals should occur only after a process has been instantiated (use _deferred_init)
# - Reorder the instantiation process:
#    - _instantiate_receiver
#    - _instantiate_sender
#    - _instantiate_function
#
#  LearningProjection requires that:
#               - _instantiate_sender and _instantiate_receiver be called in reverse order,
#               - some of their elements be rearranged, and
#               - MappingProjection.instantiate_parameter_state() be called in MappingProjection._instantiate_attributes_after_function
#               this is because:
#               - _instantiate_sender needs to know whether or not a MonitoringMechanism already exists
#                   which means it needs to know about the LearningProjection's receiver (MappingProjection)
#                   that it uses to find the ProcessingMechanism being monitored (error_source)
#                   which, in turn, means that _instantiate_receiver has to have already been called
#               - _instantiate_sender must know size of weight matrix to check compatibilit of error_signal with it
#           Error Signal "sits" in Monitoring mechanim that is the sender for the LearningProjection
#  MonitoringMechanism must implement and update flag that indicates errorSignal has occured
#           this is used by MappingProjection to decide whether to update LearningProjection & weight matrix
#
# DOCUMENT: If _validate_params is overridden:
#               before call to super()._validate_params(), params specified by user are in request_set
#               after call to super()._validate_params(), params specified by user are in target_set
# DOCUMENT: Function subclasses must be explicitly registered in Components.__init__.py
# DOCUMENT: ParameterStates are instantiated by default for any FUNCTION params
#                unless suppressed by params[FUNCTION_PARAMS][PARAMETER_STATES] = None
#           Currently, ControlProjection and LearningProjection suppress parameterStates
#                by assigning paramClassDefaults = {FUNCTION_PARAMS: {PARAMETER_STATES:None}}
# DOCUMENT: .params (= params[Current])
# DOCUMENT: requiredParamClassDefaultTypes:  used for paramClassDefaults for which there is no default value to assign
# DOCUMENT: CHANGE MADE TO FUNCTION SUCH THAT paramClassDefault[param:NotImplemented] -> NO TYPE CHECKING
# DOCUMENT: EVC'S AUTOMATICALLY INSTANTIATED prediction_mechanisms USURP terminalMechanism STATUS
#           FROM THEIR ASSOCIATED INPUT MECHANISMS (E.G., Reward Mechanism)
# DOCUMENT:  PREDICTION_MECHANISM_TYPE IS A TYPE SPECIFICATION BECAUSE INSTANCES ARE
#                 AUTOMTICALLY INSTANTIATED BY EVMechanism AND THERE MAY BE MORE THAN ONE
# DOCUMENT:  PREDICTION_MECHANISM_PARAMS, AND THUS MONITOR_FOR_CONTROL APPLIES TO ALL prediction_mechanisms
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
#    - self.value is the lingua franca of (and always) the output of an function
#           Mechanisms:  value is always 2D np.array (to accomodate multiple states per Mechanism
#           All other Function objects: value is always 1D np.array
#    - Mechanism.value is always an indexible object of which the first item is a 1D np.array
#               (corresponding to the value of Mechanism.outputState and Mechanism.outputStates.items()[0]
#     Mechanism.variable[i] <-> Mechanism.inputState.items()e[i]
#     Mechanism.value[i] <-> Mechanism.ouptputState.items()e[i]
#    - variable = input, value = output
#    - Function.variable and Function.value are always converted to 2D np.ndarray
#             (never left as number, or 0D or 1D array)
#             [CLEAN UP CODE THAT TESTS FOR OTHERWISE] - this accomodate the possiblity of multiple states,
#                 each of which is represented by a 1D array in variable and value
#             Whenever self.value is set, should insure it is a 2D np.ndarray
#             output of projections should always be 1D array, since each corresponds to a single state
#     variable AND Mechanism output specification:
#     [0, 1, 2] (i.e., 1D array) => one value for the object
#                                (e.g., input state for a MappingProjection, or param value for a ControlProjection)
#     [[0, 1, 2]] (i.e., 2D array) => multiple values for the objectn (e.g., states for a mechanism)
#     CONTEXTUALIZE BY # OF INPUT STATES:  IF ONLY ONE, THEN SPECIFY AS LIST OF NUMBERS;  IF MULITPLE, SPECIFIY EACH AS A LIST

# DOCUMENT: When "chaining" processes (such that the first Mechanism of one Process becomes the last Mechanism
#               of another), then that Mechanism loses its MappingProjection from the input_state
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
#           they must be explicitly specified using ParamValueProjection tuple: (paramValue, CONTROL_PROJECTION)
#     - Clean up ControlProjection InstanceAttributes
# DOCUMENT _instantiate_state_list() in Mechanism
# DOCUMENT: change comment in DDM re: FUNCTION_RUN_TIME_PARAM
# DOCUMENT: Change to InputState, OutputState re: owner vs. ownerValue
# DOCUMENT: use of runtime params, including:
#                  - specification of value (exposed or as tuple with ModulationOperation
#                  - role of  RuntimeParamModulationPref / ModulationOperation
# DOCUMENT: INSTANTIATION OF EACH DEFAULT ControlProjection CREATES A NEW outputState FOR DefaultController
#                                AND A NEW inputState TO GO WITH IT
#                                UPDATES VARIABLE OF owner TO BE CORRECT LENGTH (FOR #IN/OUT STATES)
#                                NOTE THAT VARIABLE ALWAYS HAS EXTRA ITEM (I.E., BEGINS AT INDEX 1)
# DOCUMENT: IN INSTANTIATION SEQUENCE:
#              HOW MULTIPLE INPUT AND OUTPUT STATES ARE HANDLED
#             HOW ITEMS OF variable AND owner.value ARE REFERENCED
#             HOW "EXTERNAL" INSTANTIATION OF States IS DONE (USING ControlProjection.instantiateSender AS E.G.)
#             ADD CALL TO Mechanism._update_value SEQUENCE LIST
# DOCUMENT: DefaultController
# DOCUMENT: Finish documenting def __init__'s
# DOCUMENT: (In Function):
                        #     Instantiation:
                        #         A function can be instantiated in one of several ways:
                        # IMPLEMENTATION NOTE:  *** DOCUMENTATION
                        # IMPLEMENTATION NOTE:  ** DESCRIBE VARIABLE HERE AND HOW/WHY IT DIFFERS FROM PARAMETER
# DOCUMENT Runtime Params:
#              INPUT_STATE_PARAMS,
#              PARAMETER_STATE_PARAMS,
#              OUTPUT_STATE_PARAMS
#              PROJECTION_PARAMS
#              MAPPING_PROJECTION_PARAMS
#              CONTROL_PROJECTION_PARAMS
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
# CONFIRM: VALIDATION METHODS CHECK THE FOLLOWING CONSTRAINT: (AND ADD TO CONSTRAINT DOCUMENTATION):
# DOCUMENT: #OF OUTPUTSTATES MUST MATCH #ITEMS IN OUTPUT OF function **

#endregion

#region PREFERENCES: ---------------------------------------------------------------------------------------------------------
#
# IMPLEMENT: make it so that specifying only setting for pref automatically assigns level to INSTANCE for that object
# IMPLEMENT: PreferenceLevel SUBTYPE
#            For Function Components:  ADD PreferenceLevel.SUBTYPE with comments re: defaults, etc.
#
# FIX:  SHOULD TEST FOR prefsList ABOVE AND GENERATE IF IT IS NOT THERE, THEN REMOVE TWO SETS OF CODE BELOW THAT DO IT
#
# FIX: Problem initializing classPreferences:
# - can't do it in class attribute declaration, since can't yet to refer to class as owner (since not yet instantiated)
# - can't use @property, since @setters don't work for class properties (problem with meta-classes or something)
# - can't do it by assigning a free-standing preference set, since its owner will remain DefaultProcessingMechanism
#     (this is not a problem for objects, since they use the @setter to reassign ownership)
# - another side effect of the problem is:
#   The following works, but changing the last line to "PreferenceLevel.CATEGORY" causes an error
#     DDM_prefs = ComponentPreferenceSet(
#                     # owner=DDM,
#                     prefs = {
#                         kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#                         kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
#                         kpRuntimeParamModulationPref: PreferenceEntry(ModulationOperation.OVERRIDE,PreferenceLevel.CATEGORY)})
#
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
#
# FIX: Add specification of setting type to pref @setter, that is passed to PreferenceSet.set_preference for validation
#
# FIX:  replace level setting?? (with one at lower level if setting is not found at level specified)
# QUESTION:  IS PreferenceSet.level attribute ever used?
#
# QUESTION:  WHAT IS THE RELATIONSHIP BETWEEN:
#                         CLASS PREFERENCES IN .__init__.py  (OMITTING THIS ALLOWS INSTANCE TO BE SPECIFIED DIRECTLY)
#                         ONES IN ComponentPreferenceSet
#                         CUSTOM SETS DEFINED AS ClassPreferences IN CLASS DECLARATION?
#
#
# IMPLEMENT: move defaults (e.g., defaultMechanism) to preferences
#
# IMPLEMENT: change pref names from name_pref to namePref
#              (rectifying whatever conflict that will produce with other names)
#endregion

#region LOG: -----------------------------------------------------------------------------------------------------------------
#
# IMPLEMENT:
#             0) MOVE LIST OF RECORDED ENTRIES TO LOG_ENTRIES param, AND USE logPref TO TURN RECORDING ON AND OFF
#             X) VALIDATE LOG VALUES (IN set_preferences)
#             Y) Fix CentralClock
#             4) IMPLEMENT RELEVANT SETTER METHODS IN Process, Mechanism and Projections (AKIN TO ONES IN State):
#                  MOVE IT TO LEVEL OF Function??
#             1) IMPLEMENT LOGGING "SWITCH" SOMEWHERE THAT TURNS LOGGING ON AND OFF: activate_logging, deactive_logging
#                 (PROCESS.pathway.prefs.logPref?? OR AT SYSTEM LEVEL?? OR AS PREF OR ATTRIBUTE FOR EVERY OBJECT?
#                IMPLEMENT THIS IN "IF" STATEMENT OF value SETTER METHODS
#                          MAKE SURE THIS CONTROLS APPENDING OF VALUES TO ENTRIES IN A CONTEXT-APPROPRIATE WAY
#             3) FINISH WORKING OUT INITIALIZATION (IN Function AND IN Log):
#                 SHOULD TRY TO GET ENTRIES FROM logPrefs??
#                 SHOULD USE classLogEntries AS DEFAULT IN CALL TO Log;
#                 SHOULD ADD SysetmLogEntries IN CALL TO Log (IN FUNCTIONS OR IN LOG??)
#                 SHOULD ADD LOG_ENTRIES PARAM IN WHICH VARIABLES CAN BE ASSIGNED
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
# - IMPLEMENT Logging of values during initialization
# - IMPLEMENT DefaultControlMechanism(object) / DefaultController(name) / SYSTEM_DEFAULT_CONTROLLER(str)
#
# - SystemDefaultInputState and SystemDefaultOutputState:
# - SystemDefaultSender = ProcessDefaultInput
# - SystemDefaultReceiver = ProcessDefaultOutput
# - DefaultController
#
# Reinstate Default state and projections and DefaultMechanism
# *** re-implement defaults:
#      In state() and Function:
#          DefaultState
#      In projection() and Function:
#          DefaultProjection
#      In Process_Base:
#          kwProcessDefaultProjection: Components.Projections.MappingProjection
#          kwProcessDefaultProjectionFunction: Components.Function.LinearMatrix
#  DefaultMechanism is now being assigned in Process;
#  -  need to re-instate some form of set_default_mechanism() in Mechanism
#
#endregion

#region TIMESCALE ------------------------------------------------------------------------------------------------------

# FIX:  TIME_STEP SHOULD HAVE A GLOBAL PARAMETER, AND THEN DDM TIME_STEP MODE USES THAT
# FIX: Put in an "apology" exception message if anything that can't handle it is called to run in time_step mode.

# IMPLEMENT:  leave default time_scale = None for mechanism
#                      (and change it back wherever it ha been assigned to TimeScale.TRIAL)
#             when process is executed, mechanism defaults to process.time_scale;
#                 if process is run on its own (i.e., not in a system), its time_scale should default to TimeScale.TRIAL
#             when system is executed, mechanism defaults to system.time_scale;
#                 if that is None, it should default to TimeScale.TRIAL
#             deal with it in the following places:
#   SYSTEM:
#     self._execute_processing(clock=clock, time_scale=time_scale, context=context)

#     def _execute_processing(self, clock=CentralClock, time_scale=TimeScale.Trial, context=None):

#     mechanism.execute(time_scale=self.timeScale,

#     def _execute_learning(self, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):

#     def execute(self,
#                 input=None,
#                 clock=CentralClock,
#                 time_scale=None,
#                 # time_scale=TimeScale.TRIAL

#     self._execute_learning(clock=clock, time_scale=time_scale, context=context + LEARNING)

#     component.execute(clock=clock,
#                       time_scale=self.timeScale,
#                       # time_scale=time_scale,
#                       context=context_str)

#     def run(self,
#             inputs,
#             num_executions=None,
#             reset_clock=True,
#             initialize=False,
#             targets=None,
#             learning=None,
#             call_before_trial=None,
#             call_after_trial=None,
#             call_before_time_step=None,
#             call_after_time_step=None,
#             clock=CentralClock,
#             time_scale=None,
#         #     time_scale=TimeScale.TRIAL,
#             context=None):
#  PROCESS:
#     def execute(self,
#                 input=None,
#                 # params=None,
#                 target=None,
#                 clock=CentralClock,
#                 time_scale=None,
#                 # time_scale=TimeScale.TRIAL,
#                 runtime_params=None,
#                 context=None

#     mechanism.execute(clock=clock,
#                       time_scale=self.timeScale,
#                       # time_scale=time_scale,

#     # Execute learningSignals

#     if self._learning_enabled:
#         self._execute_learning(clock=clock, context=context)
#         # self._execute_learning(clock=clock, time_scale=time_scale, context=context)

#     def _execute_learning(self, clock=CentralClock, context=None):

#     # def _execute_learning(self, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):
#
#  PROJECTION:
#     ControlProjection, LearningProjection, MappingProjection:

#     def execute(self, params=None, clock=CentralClock, time_scale=None, context=None):

#     # def execute(self, params=None, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):


#endregion

#region COMPONENT:
# -----------------------------------------------------------------------------------------------------------
#
# FIX!!: Get straight ComponentName vs. ComponentType (e.g.,. EVCMechanism, ComparatorMechanism)
#
# FIX!!: VARIABLE VS. VALUE BUSINESS:
#     QUESTION: is mechanism.value always == mechanism.outputValue (if not, document example)
#     QUESTION: is self.value re-initialized prior to every system execution? process execution?
#     FIX: Mechanism.inputValue and outputValue should both be lists (not np.arrays)
#     FIX: WHY BOTHER WITH inputValue ATTRIBUTE?  IF IT IS WORTH KEEPING, ADD TO DOCUMENTATION OF MECHANISM AND INPUTSTATE
#     IMPLEMENT: 7/3/16 inputValue (== self.variable) WHICH IS 2D NP.ARRAY OF inputState.value FOR ALL inputStates
#     DOCUMENTATION:  inputValue is a list, variable is a 2d np.nparray
#     FIX:  RECONCILE DOCUMENTATION WITH ACTUALITY:  value == outputValue or just 1st item of outputValue
#           CURRENTLY:  value = outputValue (DDM doesn't even have an outputValue
#     FIX:  MAKE IT SO THAT value = output of function, and outputValue is what is returned by execute
#     FIX:                  check that outputValue is concatenation of outputState values
#     IMPLEMENT .input FOR Component:  == ndarray of all inputState.variables
#     IMPLEMENT .output FOR Component:  == ndarray of all outputState.variables
#     FIX: GET STRAIGHT system.input vs. system.inputValue
#     FIX: GET STRAIGHT system.value vs. system.output vs. system.oputputValue
#     FIX: GET STRAIGHT process.input vs. process.inputValue (should be list of ProcessInputState.values)
#     FIX: CHANGE .input to .external_input
#     FIX: process.inputValue == process.variable; GET RID OF inputValue?? or replace variable with it? (AND OTHER OBJECTS?)
#     FIX: GET STRAIGHT process.value vs. system.output vs. system.oputputValue
#     FIX: IMPLEMENT .output FOR Process:  == ndarray of all outputState.variables
#                         # FIX: THESE NEED TO BE PROPERLY MAPPED
#                         return np.array(list(item.value for item in self.lastMechanism.outputStates.values()))
#     FIX: CHECK FOR dtype == object (I.E., MIXED LENGTH ARRAYS) FOR BOTH VARIABLE AND VALUE REPRESENTATIONS OF MECHANISM)
#     IMPLEMENT HIERARCHICAL SETTERS AND GETTERS FOR .value AND .metavalues;  USE THEM TO:
#                                                                             - REPRESENT OUTPUT FORMAT OF function
#                                                                             - ENFORCE 1D DIMENSIONALITY OF ELEMENTS
#                                                                             - LOG .value AND .metavalues
#     FIX: USE LIST:
#         output = [None] * len(self.paramsCurrent[OUTPUT_STATES])
#     FIX: USE NP ARRAY
#         output = np.array([[None]]*len(self.paramsCurrent[OUTPUT_STATES]))
#     FIX: MAKE CONSISTENT: self.inputValue and self.variable for process and system,
#     FIX:                  or just make inputValue a property that returns self.variable

# FIX:  ALLOW `name` TO BE A PARAM IN A PARAMS DICT (WHICH OVERRIDES name ARGUMENT OF CONSTRUCTOR)
#
# FIX: MAKE ALL OBJECT PARAMS @PRPOERTY SO THEY ARE READ-ONLY (I.E., ACCESSSIBLE ONLY VIA ASSIGN_PARAMS)
#      AND USE SETTER TO ISSUE WARNING ON ATTEMPTS TO ASSIGN;  INTERNALLY USE _PARAM TO DO DIRECT ASSIGNMENTS
#
# FIX:  IN _validate_params, ARE FUNCTION_PARAMS CHECKED AGAINST FUNCTION?  SHOULD BE.
# FIX:           MODIFY THIS TO USE user_params (STILL TREATING function_param_specs AS BELOW)
#
# INSTANTATION OF ARGS AS OBJECTS
# PROBLEM: By allowing specification of an arg to be an object,
#              but using it as a template (to recreate another instance that will actually be used)
#              preclude being able to specify a particular object.
#          This is not a problem for Function Components, for which specific instances are not needed
#              (although is inefficient: have to instantiate each twice -- particularly salient for Matlab-based ones)
#              but what about other object types (e.g., projections), that might be explicitly instantiated for
#              use in one or more places, or created in one place and used in another (e.g., projections for a Process);
#              such items should/would be usable as templates but not actual objects
#          ??SOLUTIONS:
#              - add attribute that determines whether the object should be used an instance or a template?
#                ?? which should be the default behavior?
#              - determine use by context:  items created inline for args = templates;  assigned items = instances??
#
# IMPLEMENT: is_<componentType> typespec annotation (for Function, Mechanism, State and Projection)
#
# IMPLEMENT: GET RID OF params ARG;  replace assignments as follows:
#            OLD VERSION:
#                ASSIGNMENT:
#                    params[kwSomeParam] = some_value
#                     OR
#                    params = {kwSomeParam:some_value}
#                CALL:
#                    someFunction(params=params)
#            NEW VERSION:
#                ASSIGNMENT:
#                    someParamsDict[kwSomeParam] = some_value
#                     OR
#                    someParamsDict = {kwSomeParam:some_value}
#                CALL:
#                    someFunction(**someParamsDict)
#            NOTE: THIS ONLY WORKS IF eval(kwSomeParam) (== some_param) is an arg for someFunction
#                  (i.e.:  def someFunction(some_param=some_default_value))
#                  FOR FUNCTION THAT MUST ACCEPT PARAMS NOT SPECIFIED AS ARGS, THEN INCLUDE **kwargs AS ARG
#                  AND THEN PASS kwargs TO _assign_args_to_param_dicts as params:
#                  _assign_args_to_param_dicts(params=kwargs)
#                  any entries that have keys matching an arg of someFunction will be assigned to the corresponding args
#                  any (and only those) entries in someParamsDict that have keys that don't match an arg of someFunction
#                      will be left in kwargs, and passed to assign_args_as_param_dicts() in the params dict
#
# FIX: REFACTOR Function._instantiate_function TO USE INSTANTIATED function (rather than class ref)
#      AND Function.add_args_to_param_classes:
#      RATHER THAN EXTRCTING PARAMS, CONVERTING IT INTO A CLASS AND THEN RE-INSTANTIATING IN _instantiate_function
# FIX:
#     Specification of projections arg for Process level:  projection object?  matrix??
#     kwFullConnectivity not working on outputLayer in Multilayer Learning Test Script
#     Flattening of matrix param of function arg for MappingProjection
#
# FIX / IMPLEMENT:  change .function to a property, that refers to ivar ._function;
#                  make any internal direct assignments to ._function
#                  make setter for .function that insures function_params have also been appropriately changed
#                  by comparing them with new function's user_params,
#                  deleting any that don't match (presumably from old function that are still there)
#                  and assigning defaults for new function's params that are not missing / not yet there
# IMPLEMENT / DOCUMENT:  MODIFY ``function`` ATTRIBUTE SO THAT, IF IT IS NOT A SUBCLASS OF Function
#                        (E.G., IT IS A LAMBDA FUNCTION), THEN FUNCTION_PARAMS IS DISABLED.
#                        -- SEE EXAMPLE OF transfer_mechanism_Y in Scratch Pad
#                        THEN, INTEGRATE WITH SPECIFICATION OF INPUT_STATES,
#                        BY ADDING ``index`` ATTRIBUTE TO INPUT_STATES, AND ASSIGNING EACH ONE TO RELEVANT
#                        ITEM IN ``variable`` OF CUSTOM FUNCTION.
# IMPLEMENT: REFACTOR ALL ARGUMENT/ATTRIBUTE RELATIONSHIPS AS FOLLOWS:
#      ALL ARGUMENT-RELATED ATTRIBUTES ARE IMPLEMENTED AS PROPERTIES (FOR ASSIGNMENT VALIDATION, LOGGING, ETC.):
#        THE INTERNAL VALUE IS STORED IN _argument_related_attribute
# IMPLEMENT: make paramsCurrent a @property (DONE?) and force validation on assignment if validationPrefs is set
# IMPLEMENT: @property for FUNCTION_PARAMS that parses tuple vs. direct value
#            (replace existing function in ParameterStates)
#
#  _validate_function:
#
# FIX / IMPLEMENT: Make sure that if function is reassigned (.e.g, using _assign_defaults),
# FIX:                  that function_params are changed too
#
# FIX / IMPLEMENT: "MODIFIED RUNTIME_PARAMS":
#       CURRENTLY FUNCTIONING ("NEW" - 11/27/16, AS OF 11/29/16)
#       SHOULD BE MODIFIED SO THAT FUNCTION PARAMS ARE KEPT ONLY IN, AND ACCESSED DIRECTLY FROM
#       THE FUNCTION ITSELF (function_object.user_params) USING EITHER:
#           AN @PROPERTY FOR function_params TO GET THEM or
#           POINT user_params.function_params TO THEM (rather than it being a dict)
#
# FIX: _validate_params ALWAYS ALLOW PARAMETER_STATE_PARAMS TO PASS
# FIX: get rid of is_numeric_or_none; replace throughout with tc.optional(is_numeric)
# FIX: implement _assign_defaults where flagged
#
# QUESTION: is it possible to specify a function param in a params dict if the arguments appear in the __init__
#           method?  And, in either case, does specifying function params in a params dict overrided the value
#           assigned in an explicit instantation of the function in function arg of the __init__ method?
# QUESTION: does a parameter dict have to put projection params in a PROJECTION_PARAMS subdictionary, or can it
#           it simply include entries for the params along with (i.e., at the same level as) the PROJECTION_TYPE entry
#
# IMPLEMENT: is_numeric_or_none -> optional_numeric
#
# FIX: CHANGE PROCESSING MECHANISMS TO USE update RATHER THAN execute, AND TO IMPLEMENT FUNCTION
# FIX: For SUBTYPES, change funtionType to functionSubType (may interacat with naming)
# IMPLEMENT:
#     Move code specific to _deferred_init from sublass.__init__() to Function.__init__() (MODIFIED 8/14/16 NEW)
#     PROBLEM: variable is called variable_default in Function, and params is param_defaults
#              but something different in subclasses, so not recognized; need to standardize across all classes
# IMPLEMENT: MODIFY SO THAT self.execute (IF IT IS IMPLEMENTED) TAKES PRECEDENCE OVER FUNCTION
#                 BUT CALLS IT BY DEFAULT);  EXAMPLE:  IntegratorMechanism
# IMPLEMENT:  change specification of params[FUNCTION] from class to instance (as in ControlProjection functions)
# IMPLEMENT:  change _validate_variable (and all overrides of it) to:
#              _validate_variable(request_value, target_value, context)
#              to parallel _validate_params, and then:
# IMPLEMENT: some mechanism to disable instantiating ParameterStates for parameters of an function
#                that are specified in the script
#            (e.g., for EVC.function:
#                - uses LinearCombination,
#                - want to be able to specify the parameters for it
#                - but do not need any parameterStates assigned to those parameters
#            PROBLEMS:
#                - specifying parameters invokes instantation of parameterStates
#                    (note: can avoid parameterState instantation by not specifying parameters)
#                - each parameterState gets assigned its own functions, with the parameter as its variable
#                - the default function for a parameterState is LinearCombination (using IDENTITY_MATRIX)
#                - that now gets its own parameters as its variables (one for each parameterState)
#                - it can't handle kwOperaton (one of its parameters) as its variable!
#            SOLUTION:
#                - FUNCTION_PARAMS: {PARAMETER_STATE: None}}:  suppresses ParameterStates
#                - handled in Mechanism._instantiate_parameter_states()
#                - add DOCUMENTATION in Components and/or Mechanisms or ParameterStates;
#                      include note that functionParams are still accessible in paramsCurrent[functionParams]
#                      there are just not any parameterStates instantiated for them
#                          (i.e., can't be controlled by projections, etc.)
#                - TBI: implement instantiation of any specs for parameter states provided in PARAMETER_STATES
#
# Implement: recursive checking of types in _validate_params;
# Implement: type lists in paramClassDefaults (akin requiredClassParams) and use in _validate_params
            # IMPLEMENTATION NOTE:
            #    - currently no checking of compatibility for entries in embedded dicts
            #    - add once paramClassDefaults includes type lists (as per requiredClassParams)
# Implement categories of Function functions using ABC:
# - put checks for constraints on them (e.g., input format = output format)
# - associate projection and state categories with function categories:
#    e.g.:  mapping = transform;  input & output states = aggregate
#
#endregion

#region SYSTEM ---------------------------------------------------------------------------------------------------------
#
# IMPLEMENT!! Ted's toposort
# IMPLEMENT OrderedSet for toposort execution sets
# IMPLEMENT Replace executionList with sorted_execution_list (i.e., sort once formed, so there is only one version)
# IMPLEMENT:  OUTPUT EDGE LIST FROM GRAPH
# IMPLEMENT: Add PREDICTION to list of mechanism specifications in System (and document in System, and EVCMechanism)
# **IMPLEMENT: System.monitored_output_states:
#              @property, that gets list of all outputStates monitored by the system's controller
#              object should include their names, objects, and the inputState used to monitor it
# IMPLEMENT:  INITIALIZE USING TOPOSORT AND THEN RUN WITH FULL SET OF PROJECTIONS
#          VS INITIAL STATE ATTRIBUTE IN MECHANISM_TUPLES;
#                                   FLAG SOURCES OF FEEDBACK PROJECTIONS AS NEEDING THIS SPECIFIED
#                                   INCLUDE KEYWORD "IGNORE" THAT MEANS DON'T USE THAT PROJECTION ON INITIALIZATION PASS
#
# FIX: System.mechanismList.mechanismNames
# FIX: MAKE SURE THIS IS OK (IN System):
#                                 # MODIFIED 9/15/16 NEW:
#                                 values.append(output_state.value)
# FIX:  Order target assignments for system according to order in targetMechanisms rather than process
# TEST SPECIFYING word_reading_process BEFORE color_naming_process
#
# ** FIX: FIGURE OUT HOW TO GET DILL WORKING TO CACHE SYSTEM IN System._cache_state, OR STORE AS BINARY OBJECT
#      THIS NEEDS TO BE DONE FOR EVC SIMULATIONS, SO THAT THERE IS NOT "PARAMETER" CREEP FROM RUNNING THOSE:
#      EVC simulation may leave system in different state than end of last trial:
#      NEED TO CACHE SYSTEM BEFORE SIMULATIONS, AND RESTORE AT END.
#
# FIX: CHANGE <system>.processes to <system>.process_tuples
# FIX: DEAL WITH "INITIALIZE_CYCLE" MECHANISMS IN GRAPH
# FIX: THE FOLLOWING SHOULD SPECIFY a AS BOTH ORIGIN AND TERMINAL: [a, b, a]
# FIX: *** FLAG "INTERNAL" ORIGIN MECHANISMS (I.E., ONES THAT ALSO HAVE FEEDBACK CONNECTIONS)
# FIX:  ADD SOMEWHERE
    # if self.verbosePref:
    # print('{} has feedback connections; be sure that the following items are properly initialized:'.
    #       format(self.name))
#
# QUESTION: CYCLIC SYSTEMS:
#                HOW TO HANDLE MECHANISMS WITH OUTGOING FEEDBACK PROJECTIONS:  NEED TO BE EXPLICITLY INITIALIZED
#                HOW TO HANDLE MECHANISMS THAT ARE IN TWO PROCESSES (E.G., "SEQUENTIAL" PROCESSES):
#                   should mechanism that is TERMINAL for one process but is an ORIGIN (or projects to) another
#                   be treated as an origin and/or terminal or neither?
#                   (SEE Cyclic System Test Script)
# ANSWER: Default to ignore projection on first pass
#         Allow it to use prior values between runs/executions (modulo no reset of CentralClock)
#         Allow it to be specified as a parameter
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
#                          + MappingProjection object
#                          + IDENTITY_MATRIX: len(sender.value) == len(receiver.variable)
#                          + kwFull (full cross-connectivity) [** ADD THIS AS SPEC FOR LinearMatrix FUNCTION)
#                          + timing params
#      Processes (and use their pathways)
#    Run toposort to get linear structure
#
#    EXECUTION:
#    run function:
#        Calls each Process once per time step (update cycle)
#
#    "SEQUENTIAL"/"ANALYTIC" MODE:
#    1) Call every Process on each cycle
#        a) Each Process calls the Mechanisms in its Pathway list in the sequence in which they appear;
#            the next one is called when Mechanism.receivesFromProjections.frequency modulo CurrentTime() = 0
#
# VS:
#        a) Each Process polls all the Mechanisms in its Pathway list on each cycle
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
#     Each Process calls update method of each mechanism in its pathway
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
# IMPLEMENT:  PhaseSpec:
#   - phaseSpec for each Mechanism in Process::
#        integers:
#            specify time_step (phase) on which mechanism is updated (when modulo time_step == 0)
#                - mechanism is fully updated on each such cycle
#                - full cycle of System is largest phaseSpec value
#        floats:
#            values to the left of the decimal point specify the "cascade rate":
#                the fraction of the outputvalue used as the input to any projections on each (and every) time_step
#            values to the right of the decimal point specify the time_step (phase) at which updating begins
# QUESTION: SHOULD OFF PHASE INPUT VALUES BE SET TO EMPTY OR NONE INSTEAD OF 0?
#           IN SCRIPTS AND EVCMechanism._get_simulation_system_inputs()
# FIX: Replace toposort with NetworkX: http://networkx.readthedocs.io/en/stable/reference/introduction.html
# IMPLEMENT: Change current System class to ControlledSystem subclass of System_Base,
#                   and purge System_Base class of any references to or dependencies on controller-related stuff
# IMPLEMENT: *** ADD System.controller to executionList and
#                execute based on that, rather than dedicated line in System.execute
# IMPLEMENT: *** sort System.executionList (per System.show() and exeucte based on that, rather than checking modulos
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
# FIX:     RESTORE "# # MODIFIED 6/26/16 NEW:" IN self._validate_variable
# FIX:     MAKE CORRESPONDING ADJUSTMENTS IN self._instantiate_function (SEE FIX)
#
# FIX: Output of default System() produces two empty lists
#
#endregion

#region PROCESS: -------------------------------------------------------------------------------------------------------------
#
# FIX: SOFT CLAMP and HARD CLAMP (for clamp_input option): convert SOFT_CLAMP and HARD_CLAMP to enums and test for them
#
# FIX: REPLACE Process.firstMechanism and Process.lastMechanism WITH ORIGIN AND TERMINAL mechanisms THROUGHOUT PROJECT
# FIX: *** CHANGE process.firstMechanism -> process.origin
# FIX: *** CHANGE process.lastMechanism -> process.terminal
#
# FIX: add learning_mech_tuples and learningMechanisms
#
# FIX: Identify recurrent projections, designate mechanisms as INITIALIZE_CYCLE,
# FIX:          and in implement initialization of them in execution
#           OPTIONS:  ZEROS, NO INPUT FOR FIRST PASS, OR EXPLICITLY?
# FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params
# FIX: AUTO_ASSIGN_MATRIX NOT WORKING:  FIX IN Function LinearCombination
# IMPLEMENT: AUTO_ASSIGN_MATRIX  in LinearCombination or in MappingProjection?
#                                or wherever matching referenced in Process actually gets done
# FIX: Deploy _is_mechanism_spec in validation contexts generally
#
# CONFIRM: Assignment of processInputStates when mechanism belongs to more than one process
# TEST (line 1442):
    # if params:
    #     projection.matrix = params
#
# IMPLEMENT: ?REINSTATE VALIDATION OF PROCESS AND SYSTEM (BUT DISABLE REPORTING AND RE-INITIALIZE WEIGHTS IF LEARNING)
# IMPLEMENT: Process: phases in execution
# IMPLEMENT: ProcessTuples (per MechanismTuples)
# IMPLEMENT: Process SHOULD RECOGNIZE AND CALL MonitoringMechanism(s):
#            - do pass after _deferred_init to add MonitoringMechanism(s) to mechanisms_list
#              (or do so in _deferred_init pass)
#            - ??add flag that enables/disables learning? (for use by system/EVC)??
# IMPLEMENT: set _deferred_init flag on a mechanism if any component has a delayed init
#                and use in Process to filter which ones need to be called (both for efficiency and debugging)
#
# IMPLEMENT: In Process factory method:
#                 add name arg (name=)
#                 test params (in particular, kwConfig)
#                 test dict specification
#
# IMPLEMENT: Add params to Process for projection type (default: MappingProjection) and matrix type (default: random)
#
# IMPLEMENT: Factor _instantiate_pathway so that parsing/instantiation of mechanism/projection specs
#            can also be called after _deferred_init
#
# IMPLEMENT: Process:  modify execute to take training_signal arg if LearningProjection param is set
#                      (i.e., specify its format and where it will come from -- input or projection from a mechanism)
#
# QUESTION:  Should process.execute use phases or not?
# ANSWER: yes, for realtime mode;  so, it should check phase
#
# - FIX:
#         if len(config_item) is 3:
#                     # TEST THAT ALL TUPLE ITEMS ARE CORRECT HERE
#                     pass
#
# - FIX: NEED TO DEAL WITH SITUATION IN WHICH THE SAME MECHANISM IS USED AS THE FIRST ONE IN TWO DIFFERENT PROCESSES:
#        ?? WHAT SHOULD BE ITS INPUT FROM THE PROCESS:
#           - CURRENTLY, IT GETS ITS INPUT FROM THE FIRST PROCESS IN WHICH IT APPEARS
#           - IMPLEMENT: ABILITY TO SPECIFY WHICH PROCESS(ES?) CAN PROVIDE IT INPUT
#                        POSSIBLY MAP INPUTS FROM DIFFERENT PROCESSES TO DIFFERENT INPUT STATES??
#
# - IMPLEMENT: Autolink for pathway:
#               WHAT TO DO WITH MECHANISMS THAT RECEIVE A PROJECTION W/IN THE LIST BUT NOT THE PRECEDING
#               OVERRIDE MODE:  serial projections only within the config list
#               INHERIT MODE:   mechanisms retain all pre-specified projections:
#                                  ?? check for orphaned projections? mechanisms in NO process config??
#
# - fix: how to handle "command line" execute method parameters (i.e., specified in config tuple):
#        in check args they get incorporated into paramsCurrent, but into parameterState.value's
#        combining all of them in mechanism execute method would be "double-counting"
#        - only count the ones that changed?
#        - handle "command line" params separately from regular ones (i.e., isolate in _check_args)??
#        - pass them through parameterState execute function
#              (i.e., pass them to parameterState.execute variable or projection's sender??)
# - implement:
#     - coordinate execution of multiple processes (in particular, mechanisms that appear in more than one process)
#     - deal with different time scales
#     - response completion criterion (for TIME_STEP mode) + accuracy function
#     - include settings and log (as in ControlProjection)
#
# - implement:  add pathway arg to call, so can be called with a config
#
# - implement: alias Process_Base to Process for calls in scripts
#
#
# *** DECIDE HOW TO HANDLE RUNNING OF ALL execute FUNCTIONS ON INIT OF OBJECT:
#    ?? DO IT BUT SUPPRESS OUTPUT?
#    ?? SHOW OUTPUT BUT FLAG AS INITIALIZATION RUN
#    ?? USE CONTEXT TO CONDUCT ABBREVIATED RUN??
#
# execute methods: test for kwSeparator+COMPONENT_INIT in context:
#          limit what is implemented and/or reported on init (vs. actual run)
#endregion

#region MECHANISM: -----------------------------------------------------------------------------------------------------------
#
# FIX MonitorOutputStates ISSUES:
#     FIX: GET RID OF MonitoredOutputStatesOption enum; just use keywords (also in documentation)
#     IMPLEMENT: Replace monitored_output_states tuple format (outputState or mech, exp, weight) with
#                       (outputState or mech, MonitoredOutputStatesOptions, tuple(exp, weight))
#     FIX: MAKE MONITOR_FOR_CONTROL A REQUIRED PARAM FOR System CLASS
#          ALLOW IT TO BE:  MonitoredOutputStatesOption, Mechanism, OutputState or list containing any of those
#     FIX: NEED TO SOMEHOW CALL _validate_monitored_state FOR MONITOR_FOR_CONTROL IN SYSTEM.params[]
#     FIX: CALL _instantiate_monitored_output_states AFTER instantiate_prediction_mechanism (SO LATTER CAN BE MONITORED)
#     FIX: QUESTION:  WHICH SHOULD HAVE PRECEDENCE FOR MONITOR_FOR_CONTROL default: System,Mechanism or ConrolMechanism?
#     IMPLEMENT: EVCMechanism.MonitoredOutputStates (list of each Mechanism.outputState being monitored)
#
# FIX!!: CONSOLIDATE _instantiate_parameter_states IN Mechanism AND Projection AND MOVE TO ParameterState Module
# FIX:     Function IN Projection:  (_instantiate_attributes_before_function() and _instantiate_parameter_states())
# FIX:     ?? SHOULD THIS USE _assign_defaults ??
#
# CONFIRM: VALIDATION METHODS CHECK THE FOLLOWING CONSTRAINT: (AND ADD TO CONSTRAINT DOCUMENTATION):
#          #OF OUTPUTSTATES MUST MATCH #ITEMS IN OUTPUT OF EXECUTE METHOD **
#
# TEST: MAKE SURE THAT outputValue IS GETTING SET PROPERLY
#                  (IN Mechanism.execute OR Mecchanism._update_output_states)
#
# IMPLEMENT: dictionaries for receivesFromProjections and sendsToProjections;
#            each entry is the name of an inputState or outputState;  value is tuple with...??
# IMPLEMENT: add built-in names for mechanism's InputStates and OutputStates (like ParameterStates)
# IMPLEMENT: reference to mechanism by name in pathway (look it up in Registry)
# IMPLEMENT mechanismTuple as named tuple type
# IMPLEMENT:  Mechanism:  consider adding _update_output_states() to @property method for self.value
# IMPLEMENT:  Mechanism.initialize (that sets Mechanism.value and updates Mechanism.outputStates)
#
# FIX!!: get_mech_tuple() in MechanismList only gets first mech_tuple in the list, but there could be more than one
#      check calls to get_mech_tuple() to see if that ever will pose a problem
#      Same problem for ProcessList
#
# FIX: IN _instantiate_state:
# FIX: - check that constraint_value IS NOW ONLY EVER A SINGLE VALUE
# FIX:  CHANGE ITS NAME TO constraint_value
# Search & Replace: constraint_value -> constraint_value
#
# FIX: ALLOW EXECUTE/RUN OF MECHANISM TO USE RUN-TIME PARAM
#
# - add settings and log (as in ControlProjection)
# - Fix: name arg in init__() is ignored
#
# - MODIFY add_projection
#         IMPLEMENTATION NOTE:  ADD FULL SET OF ParameterState SPECIFICATIONS
#
# IMPLEMENT: EXAMINE MECHANISMS (OR OUTPUT STATES) IN SYSTEM FOR monitor ATTRIBUTE,
#                AND ASSIGN THOSE AS MONITORED STATES IN EVC (inputStates)
#
# - IMPLEMENT: CLEAN UP ORGANIZATION OF STATES AND PARAMS
# Mechanism components:                Params:
#   InputStates      <- InputStateParams
#   ParameterStates  <- ParameterStateParams (e.g., Control Signal execute method)
#   OutputStates     <- OutputStateParams
#   self.execute              <- MechanismFunction, MechanismFunctionParams (e.g., automatic drift rate)
#
# IMPLEMENT:  self.execute as @property, which can point either to _execute or paramsCurrent[FUNCTION]
#
# - IMPLEMENTATION OF MULTIPLE INPUT AND OUTPUT STATES:
# - IMPLEMENT:  ABSTRACT HANDLING OF MULTIPLE STATES (AT LEAST FOR INPUT AND OUTPUT STATES, AND POSSIBLE PARAMETER??
# - Implement: Add StateSpec tuple specificaton in list for  INPUT_STATE and OutputStates
#        - akin to ParamValueProjection
#        - this is because OrderedDict is a specialty class so don't want to impose their use on user specification
#        - adjust _validate_params and instantiate_output_state accordingly
# - Implement: allow list of names, that will be used to instantiate states using self.value
# - Implement: allow dict entry values to be types (that should be checked against self.value)
#
# - NEED TO INITIALIZE:            kwStateValue: NotImplemented,
# - IMPLEMENTATION NOTE: move defaultMechanism to a preference (in Mechanism.__init__() or Process.__init())
# - IMPLEMENTATION NOTE: *** SHOULD THIS UPDATE AFFECTED PARAM(S) BY CALLING RELEVANT PROJECTIONS?
# -    ASSGIGN  *** HANDLE SAME AS MECHANISM STATE AND PROJECTION STATE DEFAULTS:
#                   create class level property:  inputStateDefault, and assign it at subclass level??
# - replace "state" with "mechanism_state"
# - Generalize _validate_params to go through all params, reading from each its type (from a registry),
#                            and calling on corresponding subclass to get default values (if param not found)
#                            (as PROJECTION_TYPE and PROJECTION_SENDER are currently handled)
# IN MECHANISMS _validate_function:
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
#
# In _instantiate_state (re: 2-item tuple and Projection cases):
        # IMPLEMENTATION NOTE:
        #    - need to do some checking on state_spec[1] to see if it is a projection
        #      since it could just be a numeric tuple used for the variable of a state;
        #      could check string against ProjectionRegistry (as done in _parse_projection_ref in State)
    # IMPLEMENTATION NOTE:
    #    - should create list of valid projection keywords and limit validation below to that (instead of just str)
#
# - implement:
#     Regarding ProcessDefaultMechanism (currently defined as Mechanism_Base.defaultMechanism)
#        # IMPLEMENTATION NOTE: move this to a preference (in Process??)
#        defaultMechanism = DDM_MECHANISM
#
#endregion

#region ControlMechanism / EVCMechanism / ControlSignal: -------------------------------------------------------------
#
# FIX: implement System argument for EVCMechanism
#
# FIX: Should __init__ for ControlMechanism and EVCMechanism have default_input_value argument?
#
# FIX!!: ALLOW ControlMechanism.system ASSIGNMENT TO BE DEFERRED (CHECK ONLY ON EXECUTION, or SYSTEM INSTANTATION?)
#      COORDINATE THIS WITH deferred init for ControlProjection
#      THIS SHOULD OBVIATE NEED FOR DefaultControlMechanism
#      THEN TEST EVC System Laming Validation Test with weights assigned to EVC
#
# FIX monitor_for_control ISSUES (cf monitored_output_states ISSUES UNDER MECHANISM (ABOVE))
#     FIX:  ADD monitored_output_states ATTRIBUTE TO ControlMechanism, AND THEN MAKE SURE THAT DOCSTRING REFERENCES RESOLVE
#                 TO IT RATHER THAN EVCMechanism (AS THEY CURRENTLY DO).
#     - IMPLEMENT: MONITOR_FOR_CONTROL_OPTION for individual Mechanisms (in ControlMechanism):
#            TBI: Implement either:  (Mechanism, MonitoredOutputStatesOption) tuple in MONITOR_FOR_CONTROL specification
#                                    and/or MONITOR_FOR_CONTROL in Mechanism.params[]
#                                             (that is checked when ControlMechanism is implemented
#            DOCUMENT: if it appears in a tuple with a Mechanism, or in the Mechamism's params list,
#                          it is applied to just that mechanism
#     DOCUMENT: DIFFERENCES BETWEEN EVCMechanism.inputStates (that receive projections from monitored States) and
#                                   EVCMechanism.MonitoredOutputStates (the terminal states themselves)
#
# FIX/DOCUMENT:  WHY SYSTEM: None FOR EVCMechanism AND DefaultControlMechanism [TRY REMOVING FROM BOTH]
#
# FIX: GENERATE MORE MEANINGFUL ERROR WHEN THERE ARE NO OUTPUTSTATES TO MONITOR FOR EVC
#       USE EVC System Test Script and delete CONTROL_PROJECTION for drift_rate param in DDM.__init__()
#
# FIX: IMPLEMENT _update FOR ControlMechanism (CURRENTLY OVERRIDDEN)
# FIX: EVCMecchanism prefs not settable
# FIX: Component: UNCOMMENT WHEN EVC IS GIVEN A PREF SET
#
# FIX: MAKE EVCMechanism._update_predicted_inputs MORE EFFICIENT
#
# FIX: WHICH IS CORRECT (SEBASTIAN):
#             # MODIFIED 12/4/16 OLD:
#             self._mech_tuples.extend(self._monitoring_mech_tuples)
#             # # MODIFIED 12/4/16 NEW:
#             # self._mech_tuples.extend(reversed(self._monitoring_mech_tuples))
#             # MODIFIED 12/4/16 END
#
# NOTE:  Can implement reward rate valuation by:
# - implementing reward mechanism (gets input from environment)
# - instantiating EVC with:
# params={
#     MONITOR_FOR_CONTROL:[[reward_mechanism, DDM.outputStates[_ddm_rt]],
#     FUNCTION_PARAMS:{OPERATION:LinearCombination.Operation.PRODUCT,
#                            WEIGHTS:[1,1/x]}}
#    NEED TO IMPLEMENT 1/x NOTATION FOR WEIGHTS IN LinearCombination
#
# REFACTORING NEEDED:
# ? MODIFY State._instantiate_projections_to_state TO TAKE A LIST OF PROJECTIONS AS ITS ARG
# √ ADD METHOD TO Mechanism:  _instantiate_projections_to_state:
#      default:  ADD PROJECTION TO (PRIMARY) inputState
#      optional arg:  inputState (REFERENCED BY NAME OR INDEX) TO RECEIVE PROJECTION,
#                     OR CREATE NEW inputState (INDEX = -1 OR NAME)
# ? MODIFY DefaultProcessingMechanism TO CALL NEW METHOD FROM instantiate_control_signal_channel
# - FIX: ?? For ControlMechanism (and subclasses) what should default_input_value (~= variable) be used for?
# - EVC: USE THE NEW METHOD TO CREATE MONITORING CHANNELS WHEN PROJECIONS ARE AUTOMATCIALLY ADDED BY A PROCESS
#         OR IF params[INPUT_STATES] IS SPECIFIED IN __init__()
#
# IMPLEMENT: Implement way of specifying default allocation policy for ControlSignals in system
#
#                   MAKE IT AN ARGUMENT / ATTRIBUTE OF THE EVCMECHANISM THAT IS USED IF NO OTHER IS SPECIFED
# IMPLEMENT: ADD _instantiate_input_states TO ControlMechanism AND
#
# IMPLEMENT      MOVE ASSIGNMENT OF monitor_for_control_factors TO THERE
#
# - IMPLEMENT: controlSignals attribute:  list of control signals for mechanism
#                                        (get from outputStates.sendsToProjections)
# - IMPLEMENT: controlSignalSearchSpace argument in constructor, that can be:
#                   - 2d array (each item of which is validated for length = len(self.controlSignals
#                   - function that returns a 2d array, validate per above.
#
# - IMPLEMENT: EXAMINE MECHANISMS (OR OUTPUT STATES) IN SYSTEM FOR monitor ATTRIBUTE,
#                AND ASSIGN THOSE AS MONITORED STATES IN EVC (inputStates)
#
# - IMPLEMENT: .add_projection(Mechanism or State) method:
#                   - add controlSignal projection from EVC to specified Mechanism/State
#                   - validate that Mechanism / State.owner is in self.system
#                   ?? use Mechanism.add_projection method
# - IMPLEMENT: FUNCTION_PARAMS for cost:  operation (additive or multiplicative), weight?
# - TEST, DOCUMENT: Option to save all EVC policies and associated values or just max
# - IMPLEMENT: Control Mechanism that is assigned as default with SYSTEM specification
#               ONCE THAT IS DONE, THEN FIX: IN System._instantiate_attributes_before_function:
#                                                         self.controller = EVCMechanism(params={SYSTEM: self})#
# - IMPLEMENT: ??execute_system method, that calls execute.update with input pass to System at run time?
# ? IMPLEMENT .add_projection(Mechanism or State) method that adds controlSignal projection
#                   validate that Mechanism / State.owner is in self.system
#                   ? use Mechanism.add_projection method

# IMPLEMENT: call ControlMechanism should call ControlProjection._instantiate_sender()
#                to instantaite new outputStates and Projections in _take_over_as_default_controller()
#
# IMPLEMENT: kwPredictionInputTarget option to specify which mechanism the EVC should use to receive, as input,
#                the output of a specified prediction mechanims:  tuple(PredictionMechanism, TargetInputMechanism)
#
# FIX: CURRENTLY DefaultController IS ASSIGNED AS DEFAULT SENDER FOR ALL ControlProjections IN
# FIX:                   ControlProjection.paramClassDefaults[PROJECTION_SENDER]
# FIX:   SHOULD THIS BE REPLACED BY EVC?
# FIX:  CURRENTLY, COST_FUNCTION and COST_APPLICATION_FUNCTION ARE SPECIFIED AS INSTANTIATED FUNCTIONS
#           (IN CONTRAST TO function  WHICH IS SPECIFIED AS A CLASS REFERENCE)
#           COULD SWITCH TO SPECIFICATION BY CLASS REFERENCE, BUT THEN WOULD NEED
#             CostAggregationFunctionParams and CostApplicationFunctionParams (AKIN TO functionParams)
#
# FIX: self.variable:
#      - MAKE SURE self.variable IS CONSISTENT WITH 2D np.array OF values FOR MONITOR_FOR_CONTROL
#
# DOCUMENT:  protocol for assigning DefaultControlMechanism
#           Initial assignment is to SystemDefaultCcontroller
#           When any other ControlMechanism is instantiated, if params[MAKE_DEFAULT_CONTROLLER] = True
#                then the class's _take_over_as_default_controller() method
#                     is called in _instantiate_attributes_after_function
# it moves all ControlProjections from DefaultController to itself
#
# FIX: IN ControlProjection._instantiate_sender:
# FIX 6/28/16:  IF CLASS IS ControlMechanism SHOULD ONLY IMPLEMENT ONCE;  THEREAFTER, SHOULD USE EXISTING ONE
#
# FIX: ControlMechanism._take_over_as_default_controller() IS NOT FULLY DELETING DefaultController.outputStates
#
# FIX: PROBLEM - ControlMechanism._take_over_as_default_controller()
# FIX:           NOT SETTING sendsToProjections IN NEW CONTROLLER (e.g., EVC)
#
# SOLUTIONS:
# 1) CLEANER: use _instantiate_sender on ControlProjection to instantiate both outputState and projection
# 2) EASIER: add self.sendsToProjections.append() statement in _take_over_as_default_controller()
#
#
# BACKGROUND INFO:
# _instantiate_sender normally called from Projection in _instantiate_attributes_before_function
#      calls sendsToProjection.append
# _instantiate_control_projection normally called from ControlProjection in _instantiate_sender
#
# Instantiate EVC:  __init__ / _instantiate_attributes_after_function:
#     take_over_as_default(): [ControlMechanism]
#         iterate through old controller’s outputStates
#             _instantiate_control_projection() for current controller
#                 _instantiate_state() [Mechanism]
#                     state_type() [OutputState]
#
#endregion

#region STATE: -----------------------------------------------------------------------------------------------------
#
# FIX: WHY IS THE FIRST ARGUMENT FOR A State (AND ITS SUBCLASSES) "value" RATHER THAN "variable"??
#
# FIX: OutputState:  value as arg and value as attribute are different and therefore confusing;
#                    rename reference_value??
# IMPLEMENT: full _instantiate_input_states capability per _instantiate_output_states (see ObjectiveMechanism):
#                 ??include `senders` arg (and use version of _get_monitored_states in EVC)
# IMPLEMENT: OutputState.update: INCORPORATE paramModulationOperation HERE, AS PER PARAMETER STATE
# IMPLEMENT: REPLACE INDEXING OF Mechanism.value by OUTPUTSTATES WITH NAMES OF ITEMS IN Mechanism.value
# FIX: ``value`` should not be used as the name of the variable arg for states
#              too easily confused with their ``value`` attributes;  should be switched back to ``variable``
# IMPLEMENT: ``index`` attribute for InputState, paralleling outputState (and document accordingly)
#
# IMPLEMENT: OutputStates:
    # COMMENT:
    #     OutputStates can also be added by using the :py:func:`assign_output_state <OutputState.assign_output_state>`.
    # COMMENT
#
# FIX:  Generalize solution to problem of combining projection values when they are matrices:
#       Currently solved by embedding the value of a projection to a matrix parameterState of a MappingProjection
#           in a list (see "is_matrix_mapping").  Should probably do some more general check on dimensionality
#           of value and/or coordinate this with (e.g,. specify relevant parameter for) LinearCombination function
#
# IMPLEMENT:  OutputState functions (per following from DOCUMENTATION under Mechanism docsring:
#    Usually the ``function`` of the primary outputState transfers the result of the mechanism's function to the primary
#    outputState's ``value`` attribute (i.e., its function is the Linear function with slope=1 and intercept=0).  Other
#    outputStates may use other functions to transform the result of the mechanism's function in various ways (e.g.,
#    generate its mean, variance, etc.)
#
# IMPLEMENT outputStateParams dict;  SEARCH FOR: [TBI + OUTPUT_STATE_PARAMS: dict]
# IMPLEMENT: ability to redefine primary input and output states (i.e., to be other than the first)
#
# IMPLEMENT:  ?? ADD OPTION TO OVERRIDE "LAZY UPDATING" OF PARAMETER STATES, SO THAT ANY CHANGES CAN BE SEEN IN A PRINT
#                STATEMENT AS SOON AS THEY HAVE OCCURRED)
# IMPLEMENT: ADD OPTION TO SPECIFY WHICH OUTPUT STATES (self.outputValue) TO INCLUDE IN REPORT_OUTPUT
#            (e.g., DDM)
# IMPLEMENT:  ParamsDict - > .<param>:
#             In update parameter states, assign self.param.value == parameterState[<param>].value
#                                         and use those in mechanism functions (as current value of parameters)
#             Implement same pattern for inputState and outputState dicts, so that can have: inputState.name.value
# IMPLEMENT: option to override "lazy updating" of parameterStates (and, in particular, weight matrix)
#            -> useful for debugging;  confusing to have updates not appear until next trial
# *** NEED TO IMPLEMENT THIS (in State, below):
# IMPLEMENTATION NOTE:  This is where a default projection would be implemented
#                       if params = NotImplemented or there is no param[STATE_PROJECTIONS]
#
# **** IMPLEMENTATION NOTE: ***
#                 FOR MechainismInputState SET self.value = self.variable of owner
#                 FOR MechanismiOuptuState, SET variableClassDefault = self.value of owner
#
# - State, ControlProjection and MappingProjection:
# - if "senderValue" is in **args dict, assign to variable in init
# - clean up documentation
#
         # - %%% MOVE TO State
         #  - MOVE STATE_PROJECTIONS out of STATE_PARAMS:
         #        # IMPLEMENTATION NOTE:  MOVE THIS OUT OF STATE_PARAMS IF CHANGE IS MADE IN State
         #        #                       MODIFY KEYWORDS IF NEEDED
         #    and process in __init__ (_instantiate_projections_to_state()) rather than in _validate_params
         # - if so, then correct in _instantiate_function_params under Mechanism
         # - ADD instantiate_projection akin to _instantiate_state in Mechanism
         # - ADD validate_projection() to subclass, that checks projection type is OK for state
#
## ******* MOVE THIS TO State
#                 try:
#                     from Components.Projections.Projection import ProjectionRegistry
#                     projection_type = ProjectionRegistry[param_value.projection].subclass
#                 except ValueError:
#                     raise MechanismError("{0} not recognized as reference to a projection or projection type".
#                                          format(param_value.projection))
#
# ADD HANDLING OF PROJECTION SPECIFICATIONS (IN kwStateProjection) IN State SUBCLASSES
#                  MUST BE INCLUDED IN STATE_PARAMS
#
# GET CONSTRAINTS RIGHT:
#    self.value === Mechanism.function.variable
#    self.value ===  OutputState.variable
#    Mechanism.params[param_value] === ParameterState.value = .variable
#
    # value (variable) == owner's functionOutputValue since that is where it gets it's value
    #    -- ?? should also do this in Mechanism, as per inputState:
                # See:
                # Validate self.inputState.value against variable for FUNCTION
                # Note:  this is done when inputState is first assigned,
                #        but needs to be done here in case FUNCTION is changed
    # uses MappingProjetion as default projection
    # implement Aritmetic ADD Combination Function as FUNCTION
    # implement default states (for use as default sender and receiver in Projections)
#
# *********************************************
# ?? CHECK FOR PRESENCE OF self.execute.variable IN Function.__init__ (WHERE self.execute IS ASSIGNED)
# IN OutputState:
#   IMPLEMENTATION NOTE: *** MAKE SURE self.value OF MechanismsOutputState.owner IS
#                           SET BEFORE _validate_params of MechanismsOutputState
# *********************************************
#
# FOR inputState:
#      self.value does NOT need to match variable of inputState.function
#      self.value MUST match self.param[kwExecutMethodOuptputDefault]
#      self.value MUST match owners.variable
#
#
# # IMPLEMENTATION NOTE:  *** SHOULD THIS ONLY BE TRUE OF InputState??
#         # If owner is defined, set variableClassDefault to be same as owner
#         #    since variable = self.value for InputState
#         #    must be compatible with variable for owner
#         if self.owner != NotImplemented:
#             self.variableClassDefault = self.owner.variableClassDefault
#
# QUESTION:  WHERE DOES THIS BELONG (WHERE IS InputState USED AS VARIABLE OR ASSIGNMENT SPECIFICATION)??
#            (WAS IN Initialization arguments: UNDER __init_ FOR Mechanism_Base)
#             - variable : value, InputState or specification dict for one
#                       if value, it will be used as variable (template of self.inputState.value)
#                       if State or specification dict, it's value attribute will be used

# QUESTION:  WHERE DOES THIS BELONG (WHERE ARE ParameterStates SPECIFIED FOR ASSIGNMENT)??
#            (WAS IN Initialization arguments: UNDER __init_ FOR Mechanism_Base)
            # - params : dict
            #     Dictionary with entries for each param of the mechanism subclass;
            #     the key for each entry should be the name of the param (used to name its associated projections)
            #     the value for each entry MUST be one of the following (see Parameters above for details):
            #         - ParameterState object
            #         - dict: State specifications (see State)
            #         - projection: Projection object, Projection specifications dict, or list of either)
            #         - tuple: (value, projectionType)
            #         - value: list of numbers (no projections will be assigned)

#endregion

#region PROJECTION: ----------------------------------------------------------------------------------------------------------

# TEST KEYWORD AND NAME SPECIFICATION OF projections
#      alone and w/in tuples of Mechanism, MappingProjection and Function params
#      ADD TO META TEST SCRIPT
#
# FIX: MAKE CONSISTENT HOW PROJECTIONS HANDLE VARIABLE, VALUE AND WHAT THEY RETURN
#           ??IMPLEMENT A STANDARD FUNCTION ON SUPER THAT EXECUTES FUNCTION, SETS SELF.VALUE, AND RETURNS IT?
#
# - IMPLEMENT:  WHEN ABC IS IMPLEMENTED, IT SHOULD INSIST THAT SUBCLASSES IMPLEMENT _instantiate_receiver
#               (AS ControlProjection AND MappingProjection BOTH DO) TO HANDLE SITUATION IN WHICH MECHANISM IS SPECIFIED AS RECEIVER
# FIX: clean up _instantiate_sender -- better integrate versions for MappingProjection, ControlProjection, and LearningProjection
# FIX: Move sender arg to params, and make receiver (as projection's "variable") required
# FIX:  Move marked section of _instantiate_projections_to_state(), _check_projection_receiver(), and _parse_projection_ref
# FIX:      all to Projection_Base.__init__()
# - add kwFull to specification, and as default for non-square matrices
# - IMPLEMENTATION NOTE:  *** NEED TO SPECIFY TYPE OF MECHANIMSM_STATE HERE:  SHOULD BE DETERMINABLE FROM self.Sender
# - Implement generic paramProjection subclass of Projection:
#       stripped down version of ControlProjection, that has free-floating default inputState
#       used to control execute method params on a trial-by-trial basis (akin to use of tuples in pathway)
# - Fix: name arg in init__() is ignored
#
#endregion

#region MAPPING_PROJECTION: --------------------------------------------------------------------------------------------
#
# TEST: DOES ASSIGNING A MappingProjection OR ControlProjection TO THE Matrix ParameterState OF A MappingProjection work?
#       IF NOT, MODIFY matrix_spec TO ONLY ALLOW A LEARNING_PROJECTION.
# IMPLEMENT: typecheck function for matrix
#
#endregion

#region CONTROL_PROJECTION: --------------------------------------------------------------------------------------------
#
#
# FIX: ControlProjection._instantiate_receiver has to be called before _instantiate_function (like LearningProjection)
#              since execute (called in _instantiate_function) uses self.receiver.
#              COULD CATCH IT IN EXECUTE, AND CALL _instantiate_receiver.
# FIX: make ControlProjection functions arguments in __init__, and get them out of a dictionary
#
#      controlModulatedParamValues
#
# 0) MAKE SURE THAT PROJECTION_SENDER_VALUE IS NOT PARSED AS PARAMS
#      NEEDING THEIR OWN PROJECTIONS (HOW ARE THEY HANDLED IN PROJECTIONS?) -- ARE THEWE EVEN USED??
#      IF NOT, WHERE ARE DEFAULTS SET??
# 2) Handle assignment of default ControlProjection sender (DefaultController)
#
# FIX!!: Deferred init for ControlProjections (instead of creating DefaultController and passing to EVC)
#            Coordinate this with system assignement to ControlMechanism
#
# FIX ************************************************
# FIX: controlSignal prefs not getting assigned

# Fix: rewrite this all with @property:
#
# IMPLEMENT: when instantiating a ControlProjection:
#                   include kwDefaultController as param for assigning sender to DefaultController
#                   if it is not otherwise specified
#
# IMPLEMENT:  re-work cost functions as FUNCTION_PARAMS
#
# IMPLEMENT: when instantiating a ControlProjection:
#                   include kwDefaultController as param for assigning sender to DefaultController
#                   if it is not otherwise specified
#
#  IMPLEMENT option to add dedicated outputState for ControlProjection??
#
#
# IMPLEMENTATION NOTE:  ADD DESCRIPTION OF ControlProjection CHANNELS:  ADDED TO ANY SENDER OF A ControlProjection:
    # USED, AT A MININUM, FOR ALIGNING VALIDATION OF inputStates WITH ITEMS IN variable
    #                      ?? AND SAME FOR FOR outputStates WITH value
    # SHOULD BE INCLUDED IN INSTANTIATION OF CONTROL MECHANISM (per SYSTEM DEFAULT CONTROL MECHANISM)
    #     IN OVERRIDES OF _validate_variable AND
    #     ?? WHEREVER variable OF outputState IS VALIDATED AGAINST value (search for FIX)
#
#endregion

#region LEARNING: ------------------------------------------------------------------------------------------------------
#
# TEST: DIVERGENT SYSTEM FOR LEARNING AND EVC
# TEST: In Process._execute_learning: ONLY UPDATE PROJECTIONS FROM MECHANISMS IN THE CURRENT PROCESS
# TEST: Specify learning of individual projections in a process rather than whole process: is more than one
#             ComparatorMechanism assigned?
# TEST: does specifying learning for the process over-ride any that have been explicitly specified w/o learning?
        # XXX TEST WHICH IS TRUE:  in the process [???] OR
        # XXX that have been assigned by default (but not ones created using either inline or stand-alone specification)
# TEST: setting process.input manually (ie., in a script)
# TEST: Multilayer Learning weights:
#         restore random weights
#         run >>100 trials and check convergence
# TEST:  revalidate RL in new versions
#
# IMPLEMENT: DEFERRED INSTANTIATION OF LEARNING PROJECTION (OR ANY PROJECTION??):
#            IF LEARNING PROJECTION IS ASSIGNED TO MAPPING_PROJECTION AND/OR AS PROJECTION FOR A MONITORING MECHANISM,
#            CHECK IF THIS COMPLETES THE ASSIGNMENT OF ITS SENDER AND/OR RECEIVER AND, IF SO, CALL DEFERRED_INIT
#            ONCE IMPLEMENTED, UPDATE LearningProjection DOCUMENTATION, TO REMOVE REQUIREMENT THAT DEFERRED INIT BE CALLED
#
# IMPLEMENT: learning argument and property for system, that enforces learning on all processes
# IMPLEMENT: Make Process._learning_enabled an arg that can be used to disable learning even if learning spec is provided
#
# IMPLEMENT: add keyword LEARNING as allowable spec for process and system for learning argument
#            implement by adding a learning_spec type-check function (wherever parameter_spec) if defined
# IMPLEMENT:  add notice to error if learning is not enabled for a process for which a target mechanism is not found
# IMPLEMENT: add target (or targets) as arg in system.execute()
# IMPLEMENT:  add attribute to targetMechanisms that lists terminal mechanisms with which they are associated
# IMPLEMENT:  LEARNING_PROJECTION for ProcessingMechanism;  if specified:
#             - implement self.errorSignal attribute
# IMPLEMENT: LEARNING_PROJECTION for Process:
#             - assign self.errorSignal attribute to all mechanisms
#             - assign LearningProjection to all MappingProjections
# IMPLEMENT: learning argument and attribute for System (that assigns learning to all of its processes and,
#            and raises and exception if they can't handle it
# IMPLEMENT: LEARNING_PROJECTION_PARAMS to parallel CONTROL_PROJECTION_PARAMS
# IMPLEMENT: *** IF LEARNING IS SPECIFIED FOR PROCESS, REMOVE THE NEED TO SPECIFY TARGET:
#                  AUTOMATICALLY ASSIGN IT TO BE SAME FORMAT AS OUTPUT OF TERMINAL MECHANISM:
#                IN Process: WARN BUT SET TARGET TO self.terminal.outputState
# IMPLEMENT: RL:  make Backprop vs. RL an arg for LearningProjection (that can also be used as arg for Process)
#                 _validate_function:  must be BP or RL (add list somewhere of what is supported)
#                 IMPLEMENT: MONITOR_FOR_LEARNING AS STATE SPECIFICATION (CF. LearningProjection._instantiate_sender)
#
# FIX: change errorSignal -> error_signal (but must be sure not to interfere / get confused with existing error_signal)
# FIX: MAKE SURE LEARNING PROJECTIONS ON PROCESS ARE ALWAYS ADDED AS COPIES
# FIX: [LearningProjection]:
                # FIX: ?? SHOULD THIS USE _assign_defaults:
                # self.receiver.parameterStates[MATRIX].paramsCurrent.update(weight_change_params)
# FIX: DEAL WITH "GAP" OF LearningSignals IN A PROCESS (I.E., MAPPING_PROJECTION W/O ONE INTERPOSED BETWEEN ONES WITH)
# FIX: DEAL WITH FLOATS AS INPUT, OUTPUT OR ERROR OF LearningProjection:
# FIX:       EITHER USE TYPE CONVERSION IN BP FUNCTION,
# FIX:             VALIDATE input, outout AND error IN _instantiate_sender and instantiate_reciever
# FIX:             SET CONVERSION FLAG, AND THEN PASS CONVERSION FLAG TO INSTANTIATION OF bp UTLITY FUNCTION
# FIX:       OR DO TYPE CHECKING AND TRANSLATION IN LearningProjection
# FIX:            IMPLEMENT self.input, self.output, and self.error AND ASSIGN IN instantiate sender & receiver
# FIX:            IN _instantiate_sender AND _instantiate_receiver, CHECK FOR TYPE AND, IF FLOAT,
# FIX:            POINT self.input TO @property self.convertInput, AND SIMILARLY FOR output AND error
# FIX: Default name for LearningProjection is MappingProjection class and parameter state,
#      rather than MappingProjection's actual name
#
# SYSTEM LEARNING ***************************************************************************************************
#       - TEST ASYMETRIC SYSTEM WITH LEARNING, IN WHICH ONE PROCESS TERMINATES ON AN INTERNAL MECHANISM OF ANOTHER
#       - VALIDATE THAT targetMechanisms IN BOTH PROCESS AND SYSTEM ARE PROPERLY CONSTRUCTED
#       - IMPLEMENT: REAMINING CHECKS OF TARGETES (E.G., NUMBER OF TARGET SETS == NUMBER OF INPUT SETS
#       FIX: PROCESS_DIM IS NOT THE RIGHT VALUE HERE, AGAIN BECAUSE IT IS A 3D NOT A 4D ARRAY (NO PHASES)
#       FIX: WHY DOES MSE REPORT ARRAY IN Stroop Model Learning Test Script?
#
# PROBLEM: [FIXED PER BELOW]
#    - _instantiate_sender must know error_source, to know whether or not to instantiate a monitoring mechanism;
#        this reqiures access to LearningProjection's receiver, and thus that _instantiate_receiver be called first;
#    - that means instantiating receiver before the execute method of the MappingProjection has been instantiated
#        which, in turn, means that the weight matrix has not been instantiated
#    - that is a problem for _instantiate_sender, as there is no way to validate that
#        the length of the error_signal from the LearningProjection.sender is compatible with the dim of the weight matrix
#
# IMPLEMENT: NEW DESIGN:
#
# 0) Make sure MappingProjection from terminal Mechanism in Process is to ComparatorMechanism using IDENTITY_MATRIX
#    In System terminal mechanism search, don't include MonitoringMechanisms
#
# 1) LearningProjection:
#    - _instantiate_receiver:
#        - MappingProjection
#    - _instantiate_sender:
#        - examine mechanism to which MappingProjection project (receiver) projects:  self.receiver.owner.receiver.owner
#            - check if it is a terminal mechanism in the system:
#                - if so, assign:
#                    - ComparatorMechanism ErrorMonitoringMechanism
#                        - ProcessInputState for ComparatorMechanism (name it??) with projection to target inputState
#                        - MappingProjection from terminal ProcessingMechanism to LinearCompator sample inputState
#                - if not, assign:
#                    - WeightedErrorMechanism ErrorMonitoringMechanism
#                        - MappingProjection from preceding ErrorMonitoringMechanism:
#                            preceding processing mechanism (ppm):
#                                ppm = self.receiver.owner.receiver.owner
#                            preceding processing mechanism's output projection (pop)
#                                pop = ppm.outputState.projections[0]
#                            preceding processing mechanism's output projection learning signal (popls):
#                                popls = pop.parameterState.receivesFromProjections[0]
#                            preceding ErrorMonitoringMechanism (pem):
#                                pem = popls.sender.owner
#                            assign MappingProjection from pem.outputState to self.inputState
#                        - Get weight matrix for pop (pwm):
#                                pwm = pop.parameterState.params[MATRIX]
#    - update: compute weight changes based on errorSignal received rom ErrorMonitor Mechanism and pwm
#
# 2) ErrorMonitoring Mechanism:
#    - get MappingProjection from source of errorSignal:
#        last one (associated with terminal ProcessingMechanism) gets it from external input
#        preceding ones (associated with antecedent ProcessingMechanisms in the Process) get it from
#            the ErrorMonitor associated with the next ProcessingMechanism in the process:
#    - get weightMatrix for the output of its associated ProcessingMechanism
#        last one:  this should be identityMatrix (for MappingProjection from terminal mechanism to ComparatorMechanism)
#        preceding ones: get from self.receiver.owner.outputState.projections.params[MATRIX]
#    - ErrorMonitoring Mechanism computes the error for each element of its variable ("activation vector"):
#        last one (LinearCompartor) simply computes difference between its two inputs (target and sample)
#        preceding ones compute it as the dot product of its input (errorSignal) and weightMatrix
#    - outputState (errorSignal) has two projections:
#         one MappingProjection to the preceding ErrorMonitorMechanism
#         one LearningProjection to the output MappingProjection of its associated ProcessingMechanism
#
#
# 3) Update:
#    ?? add to System?
#    ?? use toposort?
#    ?? coordinate with updating for Mechanisms?
#
# Two object types:
# 1) ComparatorMechanism (MonioringMechanism):
#     - has two inputStates:  i) system output;  ii) training input
#     - computes some objective function on them (default:  Hadamard difference)
#     - default ComparatorMechanism that is associated with default LearningProjection
#
# 2) LearnningSignal (Projection):
#     - sender:  output of Monitoring Mechanism
#         default: receiver.owner.outputState.sendsToProjections.<MonitoringMechanism> if specified,
#                  else default ComparatorMechanism
#     - receiver: MappingProjection parameterState (or some equivalent thereof)
#
# Need to add parameterState to Projection class;  composition options:
#    - use ParameterState
#    - extract core functionality from ParameterState:
#        make it an object of its own
#        ParameterState and Training Projection both call that object
# MappingProjection should have LEARNING_PARAM which:
#    - specifies LearningProjection
#    - uses self.outputStates.sendsToProjections.<MonitoringMechanism> if specified
#    - otherwise defaults to LinearCompartor (which it instantiates for itself) and LearningProjection with BP
#
# Projection mechanism:
# Generalized delta rule:
# weight = weight + (learningRate * errorDerivative * transferDerivative * sampleSender)
# for sumSquared error function:  errorDerivative = (target - sample)
# for logistic activation function: transferDerivative = sample * (1-sample)
# NEEDS:
# - errorDerivative:  get from FUNCTION of ComparatorMechanism
# - transferDerivative:  get from FUNCTION of Process Processing Mechanism
#
# LearningProjection instantiation
# QUESTION: which should be the sender for final LearningProjection in a Process (and compute the initial errorSignal):
#             - a MonitoringMechanism to which the output (terminal) layer projects
#                  ADVANTAGES:
#                    - modular, consistent with PNL "philosophy"
#                  PROBLEMS:
#                    - the MonitoringMechanism masks the output layer as the terminal mechanism of the Process
#             - the output (terminal) layer of a process
#                  in this case, the targetMechanism would receive a projection from the output layer,
#                     and project the errorSignal back to it, which would then be assigned to outputLayer.errorSignal
#                  ADVANTAGES:
#                    - keeps the errorSignal exclusively in the ProcessingMechanism
#                  PROBLEMS:
#                    - overspecialization (i.e., less modular)
#                    - need to deal with recurrence in the System graph
#                    - as above, the MonitoringMechanism masks the output layer as the terminal mechanism of the Process
#             - output layer itself (i.e., make a special combined Processing/MonitoringMechanism subclass) that has
#                  two input states (one for processing input, another for training signal, and a targetMechanism method)
#                  ADVANTAGES:
#                    - more compact/efficient
#                    - no recurrence
#                    - errorSignal resides in ProcessingMechanism (as with all other levels)
#                    - leaves the output layer is the terminal mechanism of the Process
#                  PROBLEMS:
#                    - overspecialization (i.e., less modular)
#                    - needs additional "function" (targetMechanism function)
#            IMPLEMENTED: MonitoringMechanism
#
# IMPLEMENT: LEARNING IN Processes W/IN A System; EVC SHOULD SUSPEND LEARNING DURING ITS SIMULATION RUN
# IMPLEMENT: Recurrent (for WM in RLPM model)
# IMPLEMENT: RL (vs. BP):
#                0) Linear layer as penultimate layer (one for which output weights will be modified);
#                       (note: slope gets parameterState that is controlled by learning_rate of LearningProjection)
#                1) Use Softmax as final output layer
#                2) ComparatorMechanism:  constrain len(COMPARATOR_SAMPLE) = len(COMPARATOR_TARGET) = 1
#                          (rather than len(terminalMechanism.outputState)
#                3) FullConnectivity MappingProjection from terminalMechanism->ComparatorMechanism
#                4) LearningProjection.learningRate sets slope of Linear layer
#                ----------------
#
#                REVISED VERSION:
#
#                0) Inputs (stimuli, actions) project to expected reward using identity matrix
#                1) Softmax on expected reward array
#                2) Pick one element probabilistically [IMPLEMENT] and use that one to set output of expected reward:
#                    - calculate cumulative sum (in order of options);
#                    - then draw random num (from uniform distribution),
#                    - pick first one whose cum sum is above the random number
#                    Note: NOT expected utility; i.e., softmax is just a decision rule, not a probability estimator
#                3) Compare that reward with the one received
#                4) Update the reward prediction (input weights) of the chosen action only (not the others)
#
#                 Other versions:
#                 one in which the reward goes to infinity (how do to that?)
#                 one in which probability of softmax is learned - but isn’t that what is happening here?
##endregion

#region FUNCTION: ---------------------------------------------------------------------------------------------------
#
# FIX: name of Functions is being assigned to Type rather than subtype
# FIX: MAKE SURE REORDERING OF TESTING OF MATRIX SPEC IN LinearMatrix._validate_params IS OK
#
# IMPLEMENT BOTH FULL_CONNECTIVITY_MATRIX AND 2D np.array AND np.matrix OBJECTS
# IMPLEMENT:  Demos of Functions that plots each Function
#                                (use new "demoRange" attribute that specifies range of inputs for Function for demo)
# IMPLEMENT: Add scale to TransferFunction (but make sure it doesn't conflictf with or cause trouble for range)
# IMPLEMENT:  Add owner to all error messages in Functions
# IMPLEMENT name arg to individual functions, and manage in __init__()
# IMPLEMENT abstract Types (aggregate, TransferMechanism, tranform, objective)
# IMPLEMENT subtypes of above
# IMPLEMENT:  shortcircuit LinearCombination and Linear and LinearMatrix if params => identity
# LinearMatrix:
#   IMPLEMENTATION NOTE: Consider using functionOutputTypeConversion here
# IMPLEMENT: Consider implementing ParameterStates for Functions, and then assigning ControlProjections to them directly
#
# IMPLEMENT: RANDOMIZATION OF INITIAL WEIGHTS IN kWMatrix:
#            implement ability to specify function for randomization of weights (distribution, range, etc.)
# IMPLEMENT:
#     IN LinearCombination WEIGHTS PARAM:  */x notation:
#         Signifies that item to which weight coefficient applies should be in the denominator of the product:
#         Useful when multiplying vector with another one, to divide by the specified element (e.g., in calcuating rates)
#      EXAMPLE:
#           WEIGHTS = [1, 1/x]   [1, 2/x]
#           variable =  [2, 100]   [2, 100]
#           result:     [2, .01]   [2, 0.2]
#
# IMPLEMENT: simple Combine() or Reduce() function that either sums or multiples all elements in a 1D array
# IMPLEMENT:  REPLACE INDIVIDUAL FUNCTIONS WITH ABILITY TO PASS REFERENCE TO NP FUNCTIONS (OR CREATE ONE THAT ALLOWS THIS)
# IMPLEMENT .keyword() FOR ALL FUNCTIONS (as per LinearMatrix);  DO SAME FOR Enum PARAMS??
#
# LinearCombination:
#     Fix Finish fixing LinearCombination:
#          (checking length of 1D constituents of 2D variable);
#          confirm that for 2D, it combines
#          consider doing it the other way, and called by projections
#     Fix: ??Enforce 2D for parameters values:
#     Fix:  DOCUMENT:
#           - If its a 1D vector, then just scale and offset, but don't reduce?
#           - So, the effect of reduce would only occur for 2D array of single element arrays
#
#endregion

#region RUN: ----------------------------------------------------------------------------------------------------
#
# IMPLEMENT!!: FOR SYSTEM AND PROCESS:
#              learning OPTION in run()
#              train():  buffers and then sets enableLearning; returns error;  requires target(s)
#              test():  buffers and then unsets enableLearning; returns error
#              run():  returns outputValues
#              construct_targets():
#
# IMPLEMENT:
#     TRIAL: if verbose report number of trials for run()
#     CYCLE WARNING:  if verbose, on run() warn about any un-initialized recurrent projections
#     Equivalent of run() for initialize()
#
# IMPLEMENT: show function for results of system.execute (integrate with system.outputValues) and process.execute
# IMPLEMENT: show function for system and process that shows inputs
# IMPLEMENT: ??change specification of inputs in construct_inputs to name of process rather than mechanism
# IMPLEMENT: help function for process.run and system.run that explains required structure of inputs
#
# TEST!!: function format for inputs in Run
#
# FIX: WHAT HAPPENS IF NO TARGETS ARE PROVIDED TO RUN OR EXECUTE BUT LEARNING IS ENABLED?
#
# FIX: ADD ERROR MESSAGE IF MECHANISM REFERENCED IN KEY OF TARGET DICT (FOR run()) IS AN INTERNAL MECHANISM VS. MISSING
#
# TEST run(inputs) dict format FOR SITUATION IN WHICH TWO PROCESSES HAVE THE SAME INPUT,
#                                                     OR ONE PROCESS BRANCHES OUT FROM ANOTHER
#                  ISSUE IS WHETHER THE RIGHT NUMBER OF INPUTS ARE ASSIGNED, AND WHETHER SYSTEM KNOWS NOT TO
#                  CREATE AN INPUT FOR THE BRANCHING PROCESS (SEE RUN line 688 and SYSTEM line 1388
# FIX: ADD SUPPORT FOR RUNTIME PARAMS TO RUN
#
# FIX: Add error message if input.value is None on execute
#
# FIX: If reset_clock and/or initialize == True, set object.result = []
#
# IMPLEMENT: Use Structured array for System input (using namedtuples for inputs to each origin mechanism):
#    http://docs.scipy.org/doc/numpy/user/basics.rec.html
#    programmatically constuct named tuples?
#    tuple(n if i == k else 1 for i in range(m))
#    Result = namedtuple('Result', ['x', 'y'])
#    result = Result(5, 6)
#
#  EXAMPLES:
# my_input_layer = TransferMechanism(default_input_value=[0,0,0], function=Linear)
# my_hidden_layer = TransferMechanism(default_input_value=[0,0,0], function=Logistic)
# my_decision_layer = DDM(default_input_value=[0], function=BogaczEtAl)
#
# IMPLEMENT run Function (in Utilities.py):
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
#        function is LinearCombination (since it is a param) so function outPut is a single value
#        Need to suppress execute method, or assign some other one (e.g., CombineVectors)
#
#endregion

#region DDM_MECH: ------------------------------------------------------------------------------------------------------
#
# FIX!!: CHECK WHETHER DDM STILL HANDLES runtime_params DIFFERENTLY
# FIX: DDM -> DDMMechanism
# FIX: CLEAN UP PROBABILITY_UPPER_THRESHOLD ETC.
# FIX: combine paramsCurrent with executeParameterState.values, or use them instead??
# FIX:  move kwDDM_AnalyticSolution back to FUNCTION_PARAMS and adjust validation to allow non-numeric value
#
# TEST: test DDM with drift_rate specified as lambda function
#
# IMPLEMENT!! DDM: noise function for TIME_STEP mode (to use non-Gaussian distributions), or implemente JumpDM
# IMPLEMENT!!: ADD PARAM TO DDM THAT SPECIFIES PRIMARY INPUTSTATE (i.e., DRIFT_RATE, BIAS, THRSHOLD)
#
# IMPLEMENT: add options to multiply or fully override parameterState.values
# IMPLEMENT: time_step and terminate()
# IMPLEMENT: "MULTIPROCESS DDM" (search for this for places to implement/document)
# IMPLEMENT: "MULTIPROCESS DDM" (search for this for places to implement/document)
# IMPLEMENT: ``average_output_states`` ARGUMENT / OPTION AFTER IMPLEMENTING MULTIPROCESS DDM
# IMPLEMENT: ADD PARAM TO DDM (AKIN TO kwDDM_AnayticSolution) THAT SPECIFIES PRIMARY INPUTSTATE (i.e., DRIFT_RATE, BIAS, THRSHOLD)
# IMPLEMENT: customizable noise distribution for TIME_STEP mode??
# IMPLEMENT: interrogation protocol:  ER (mass of distribution to left and right of decision variable)
# IMPLEMENT: compute variance of path in time_step mode and report in RT_CORRECT_VARIANCE?? (but not just correct?)
# -  Clean up control signal params, modulation function, etc.
#        1) value field is initialized with self.value
#        2) value points to mechanism.outputState.value
#        3) params field is populated with list of params from paramsCurrent that are StateParams
#        4) duration field is updated at each time step or given -1
#    Make sure paramCurrent[<kwDDMparam>] IS BEING PROPERLY UPDATED (IN PROCESS?  OR MECHANISM?) BEFORE BEING USED
#                            (WHAT TOOK THE PLACE OF get_control_modulated_param_values)
#
#endregion

#region TransferMechanism ---------------------------------------------------------------------------------------------
#
# FIX: TransferMechanism:
# FIX:     - implement initial_state
# FIX:     - add equation for rate argument
#
# #endregion

#region ComparatorMechanism -----------------------------------------------------------------------------------
# FIX: IN ComparatorMechanism _instantiate_attributes_before_function:  USE ASSIGN_DEFAULT
# IMPLEMENT: ComparatorMechanism Processing Mechanism TYPE, ComparatorMechanism SUBTYPE
#endregion

#region ObjectiveMechanism -----------------------------------------------------------------------------------
#
#     Validate ObjectiveMechanism.monitor argument:
#         Note: parsing/validation of monitored_output_states (in EVCMechanism._get_montiored_states) and
#               monitor (in ObjectiveMechanism._validate_monitored_states) needs to be handled in a more principled way
#               either in their _validate_params method, or in class function
#
#     Make sure add_monitored_value works
#     Allow inputStates to be named (so they can be used as ComparatorMechanism)
#     Move it to ProcessingMechanism
#  Replace ComparatorMechanmism with ObjectiveMechanism
#   using a particular function and named inputStates
#   FIX: typechecking
#   FIX: rename `monitor` and `names` args
#   - IMPLEMENT call to _instantiate_input_states (not plural) once that is implemented (see State above):
#                    - parse `monitor` arg into inputState specifications and pass to _instantiate_input_states()
#   - IMPLEMENT TransferMechanism.outputStates[DERIVATIVE] (per Kristin)
#    Make sure it checks for multiple MappingProjections from its error_source, and that only uses those projections
#         that go to another ProcessingMechanism that itself projects to an ObjectiveMechanism (i.e., to avoid
#         ones that go to mechanisms that are not part of learning (e.g., other Processing or Control mechanisms)
#
#endregion

#region AdaptiveMechanisms -----------------------------------------------------------------------------------
#  These chnage the parameters of other mechanisms (Control) or projections (Learning)
#  Create as Type of Mechanism (after removing MonitoringMechanism and ControlMechanism
#  Move LearningMechanism and ControlMechanism under this category;  Get rid of MonitoringMechanism
#endregion

#region EVCMechanism -----------------------------------------------------------------------------------
#     Validate that EVCMechanism.inputState matches outputState from EVCMechanism.monitoring_mechanism
#     Allow it to take monitoring_mechanism as an argument
#           (in which case it must be validated, but then don't bother to instantiate ObjectiveMechanism)
#     Make sure add_monitored_value works:
#           Needs to call ObjectiveMechanism.add_monitored_value
#           Needs to update self.system.graph to include ObjectiveMechanism:
#endregion