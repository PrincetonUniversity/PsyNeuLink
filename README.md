Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


# PsyNeuLink

## ARCHITECTURE OVERVIEW
### Theoretical Architecture / Processing Hierarchy
#### Purpose:
    Provide a language/framework/toolkit for implementing models/theories of mind/brain function
    Do so, by expressing an information processing system (*everything* is a function) that:
     - is computationally general
     - adheres as closely as possible to the insights and design principles that have been learned in CS
         (e.g., function-based, object-oriented, etc.)
     - expresses (the smallest number of) "commitments" that reflect fundamental (universal) principles
         how the brain/mind is organized/functions, without committing to any particular model or theory
     - expresses these commitments in a form that is powerful, easy to use, and familiar to behavioral scientits
     - allows models of this to be implemented in as flexible a way as possible,
         in terms of architecture and functional forms, but also in the
         mix of commitments made in different parts of the system to:
         - time-scale of function
         - granularity of representation/function
     - thus, encourages users to "think" about processing in a "mind/brain-like" way,
         and yet impose no constraints on what they implement or ask their model to do

#### System

     Set of processes, made up of chains of mechanisms connected by projections, and
     managed by a budget of control signals (that control the mechanisms of each process)

     Each process is defined by a configuration that is single-threaded and executed in sequence;  however:
     Processes can overlap at the systems level
     What “executed in sequence” means depends on time-scale
         trial = truly sequential
         time_step (“realtime”) = cascaded

     - Process 
         Function that takes an input, processes it through an ordered list of mechanisms (and projections)
         and generates an output

         - Mechanism 
             Function that converts an input state representation into an output state representation
             Parameters that determine its operation, under the influence of projections

         - Projection 
             Function that takes a source, possibly transforms it, and uses it to
             determine the operation of a mechanism;  three primary types:

             + Mapping
                 Takes output state of sender, possibly transforms it,
                     and provides it as input state to receiver

             + ControlSignal
                 Takes an allocation (scalar), possibly transforms it,
                 and uses it to modulate the internal parameter of a mechanism

             + GatingSignal
                 Takes a source, possibly transforms it, and uses it to
                 modulate the input and/or output state of a mechanism

             + Learning
                 Takes an input from objection function (Utility)
                 and modulates params (e.g., weights) of projection execute method
                 + Vectorial: modifies mapping projections
                 + Evaluative: modifies control projections

## SOFTWARE ARCHITECTURE

 PsyNeuLink package dependencies:
 
    toposort.py - graph analysis
    mpi4py.py  - Parallelization using MPI
    wfpt.py - Wald first passage times for DDM

 Python Object Classes (and Initialization Arguments):  

     CLASS HIERARCHY:

     Format:
     - Class(required_arg, [optional_args]) # comment

     Hierarchy:
     - Function: abstract class - cannont be instantiated directly
         - Category: abstract classes - cannont be instantiated directly
             - Type: can be instantiated
                 <instances>

     Function(variable, params, name, prefs, context)
         Process_Base([default_input, params, name, prefs, context]) # sequence of mechanisms to execute
         Mechanism_Base([variable,                                   # default: DDM
                         params,
                         name,
                         prefs,
                         context])
             ProcessingMechanism
                 DDM([default_input,                                     # default mechanism
                      params,
                      name,
                      prefs])
                 [TBI: PDP]
             MonitoringMechanism
                 LinearComparator
             ControlMechanism
                 DefaultControlMechanism
                 EVCMechanism
         State_Base(owner_mechanism,
                        [value, params, name, prefs, context, **kargs])
             InputState(owner_mechanism,
                                [reference_value, value, params, name, prefs])
                                                                            # input to mechanism execute method
             ParameterState(owner_mechanism, [reference_value, value, params, name, prefs])
                                                                            # param values for mechanism execute method
             OutputState(owner_mechanism, [reference_value, params, name, prefs])
                                                                            # output from mechanism execute method
         Projection_Base(receiver, [sender, params, name, prefs, context])
             Mapping([sender, receiver, params, name, prefs])                    # outputState -> inputState
             ControlSignals([allocation_source, receiver, params, name, prefs])  # outputState -> parameterState
             [TBI: - Gating()]                                                   # outputState -> inputState/outputState
             [TBI: - Learning()]                                                 # outputState -> projection
         Utility_Base(variable_default, param_defaults, [name, prefs, context])
             Contradiction([variable_default, param_defaults, prefs, context])    # example function
             [TBI: Implement as abstract Type: Aggretate
                 LinearCombination([variable_default, param_defaults, prefs, context])   # combines values/vectors
                 [TBI: Polynomial()]
             [TBI: Implement as abstract Type:  Transfer() # converts values/vectors
                 Linear([variable_default, param_defaults, prefs, context])       # returns linear transform of variable
                 Exponential([variable_default, param_defaults, prefs, context])  # returns exponential transform of var.
                 Integrator([variable_default, param_defaults, prefs, context])   # returns accumulated value of variable
                 LinearMatrix([variable_default, param_defaults, prefs, context]) # maps var. to output using wt. matrix
             [TBI: Implement as abstract Type: Distribution() # generates values/vectors
             [TBI: Implement as abstract Type: Objective # evaluates performance of mechanism)]


     MODULES:

                 CLASS:                                               MODULE:

     Function(Object)............................................[PsyNeuLink.Functions.Function]

         System(Function)........................................[PsyNeuLink.Functions.ShellClassses]
             System_Base(System).................................[PsyNeuLink.Functions.System]

         Process(Function).......................................[PsyNeuLink.Functions.ShellClassses]
             Process_Base(Process)...............................[PsyNeuLink.Functions.Process]

         Mechanism(Function).....................................[PsyNeuLink.Functions.ShellClasses]
             Mechanism_Base(Mechanism)...........................[PsyNeuLink.Functions.Mechanisms.Mechanism]
                 DefaultProcessingMechanism_Base(Mechanism_Base).....[PsyNeuLink.Functions.Mechanisms.Mechanism]
                 DDM(Mechanism_Base).............................[PsyNeuLink.Functions.Mechanisms.DDM]
                 DefaultControlMechanism(Mechanism_Base)...[PsyNeuLink.Functions.Mechanisms.Mechanism]

         State(Function)................................[PsyNeuLink.Functions.ShellClasses]
             State_Base(State).................[PsyNeuLink.Functions.States.State]
                 InputState(State_Base)........[PsyNeuLink.Functions.States.InputState]
                 OutputState(State_Base).......[PsyNeuLink.Functions.States.OutputState]
                 ParameterState(State_Base)....[PsyNeuLink.Functions.States.ParameterState]

         Projection(Function)....................................[PsyNeuLink.Functions.ShellClasses]
             Projection_Base(Projection).........................[PsyNeuLink.Functions.Projections.Projection]
                 Mapping(Projection_Base)........................[PsyNeuLink.Functions.Projections.Mapping]
                 ControlSignal(Projection_Base)..................[PsyNeuLink.Functions.Projections.ControlSignal]

         Utility(Function).......................................[PsyNeuLink.Functions.ShellClasses]
             Utility_Base(Utility)...............................[PsyNeuLink.Functions.Utility]
                 Contradiction(Utility_Base).....................[PsyNeuLink.Functions.Utility]
                 LinearCombination(Utility_Base)........................[PsyNeuLink.Functions.Utility]
                 Linear(Utility_Base)............................[PsyNeuLink.Functions.Utility]
                 Exponential(Utility_Base).......................[PsyNeuLink.Functions.Utility]
                 Integrator(Utility_Base)........................[PsyNeuLink.Functions.Utility]
                 LinearMatrix(Utility_Base)......................[PsyNeuLink.Functions.Utility]


     Requirements:

     - Projection subclasses must see (particular) State subclasses in order to assign kwProjectionSender
     - State subclasses must see (particular) Projection subclasses in order to assign kwProjectionType
     - Process must see Mechanism subclasses to assign PsyNeuLink.Functions.DefaultMechanism
     - Would like Mechanism, Projection (and possible State) classes to be extensible:
         developers shoud be able to create, register and refer to subclasses (plug-ins), without modifying core code

## FORMATTING STANDARDS

     Naming Conventions:
         - class names use camelCase with an initial capitilization: ClassName
         - publicly relevant attributes use camelCase:  self.someThing
         - local variables and method args use underscores:  some_thing
         - function names use underscores (enforced by PEP8): some_method

     Itemization:
        attributes are itemized without a marker
        - arguments are itemized with dashes
        + parameters (in dicts) are itemized with pluses
        - methods are itemized with bullets
        * notes are itemized with asterisks

     Module organization:
        Imports
        Keywords
        Constants & Structures
        PrefenceSet
        Registry
        Log
        Error
        Variables
        Factory method
        Class definition

     Documentation:
         Description
         [Subclasses]
         Instantiation
         Initialization arguments:
             - variable (<type>): <description>
             - params (<type>): <description>
                 + kw<RemainderOfParamName> (type): <description>
             - name (<type>): <description>
             - prefs (<type>): <description>
             - context (<type>): <description>
         <Class>Registry:
         Naming
         Execution
         Class attributes (implemented at clas level)
         Class methods (implemented at class level)
         Instance attributes (implemented in __init__():
         Instance methods (implemented under class)

## DESIGN PATTERNS / PRINCIPLES

### Functions:
     Everything is a function
     Every call (for both initialization and execution) has five standard arguments
     - variable (value):
         as an arg in __init__, formats and establishes a default for the variable
         as an arg in a function call, serves as the input to the function
     - params (dict):
         as an arg in __init__, instantiates and establishes instance-specific defaults for function parameters;
         as an arg in a function call, used to override instance defaults for that call only
     - name (str): used to the name the function:
         function names have three levels (separated by spaces):  category, class, and instance
         classes index default instance names or if an existing name is provided
     - prefs (dict):
         contains user PreferenceSet (settings and logging)
     - context (str):
         used to license initialization calls to abstract super classes (by legitimate subclasses)
     Every instance has an execute method, that is referenced either by params[kwExecuteMethod] param OR self.execute
         this is the function that is called when executing the class to which the instance belongs:
         Process: executes the list of mechanisms in its configuration
         Mechanism:  executes the States instantiating its inputState, params, and outputState
         MechanismClass: executes each projection for which it is a receiver, and aggregates them if there are several
         Projection: translates the value of its sender and provides it for use as the value of its receiver
     Every subclass of function MUST either:
         - reference a function in paramClassDefauts[kwExecuteMethod] OR
         - implement its own method that then MUST be called "execute" (i.e., <class>.execute);
         kwExecuteMethod takes precedence (i.e., supercedes any subclass implementation of self.execute)
         kwExecuteMethod can be either:
             a reference to an instantiated function, or
             a class of one (in this case, it will be instantiated using kwExecuteMethodParams if provided)
         if a valid kwExecuteMethod is instantiated, self.execute will be aliased to it
         if kwExecuteMethod is missing or invalid, it will be assigned to self.execute
         if neither exists, an exception is raised
         NOTE:
             * As described above, the execute method of a class is referenced (and can be called) by both
                self.execute and self.paramsCurrent[kwExecuteMethod]:
                - this is done for convenience (self.execute) and flexibility (self.paramsCurrent[kwExecuteMethod])
                - when executing the function, it is generally safer and a best practice to call <instance>.function
     validate_* methods are called (usually in super.__init__) before any instantiate_* methods
         validate_* methods perform a (syntactic) check to:
             - determine if required items are present
             - deterimne if items are of the correct type relative to instance or class default
             - assign defaults where appropriate for invalid entries, with warning if in VERBOSE mode
             - NOT whether items are compatible with other entities (i.e., it is not a "semantic" check)
         instantiate_* methods perform a (semantic) check to:
             - determine if item is compatible with others
                 (e.g., variable or output of one is compatible with value of another)

     self.variable is sometimes yoked/aliased to other attributes (for semantic reasons);  for example:
         variable -> value (for States)
                  -> input (for Mechanism and Process)
     param values can, in some cases, be specified as numbers, but will be converted to a single-item list
             as the "lingua-franca" for variables
             (which they are, for the receiver's inputState function)

### States and Projections:

     - Every mechanism has three types (subclasses) of State associated with it:
         - a single InputState:
              its value serves as the input to the mechanism
              it receives one or more Mapping Projections from other mechanisms
         - one or more ParameterStates:
             their values serve as the parameters of the mechanism's kwExecuteMethod (self.execute),
             each of which receives typically one (but possibly more) ControlSignal projections
         - a single OutputState:
              its value serves as the output to the mechanism,
              and is typically assinged as the sender for other mechanisms' Mapping Projection(s)
     - State:
         every instance of State has a single value attribute (that represents its "state"; = self.variable)
         every instance of State must be explicitly assigned an existing <state>.owner (Mechanism)
         default projections can be implemented for a state (using <state>.defaultProjectionType);
             if their sender is not specified, a default one will be created (see Projection below)
         <state>.receivesFromProjections is consulted when a State's update function is executed,
             and <state>.value is updated based on those
         subclasses must implement defaultProjectionType
     - Projection:
         every projection must be explicitly assigned an existing <projection>.receiver (State)
         default states can be implemented for a projection's sender (using paramsCurrent[kwProjectionSender])
         subclasses must implement paramClassDefaults[kwProjectionSender]

     Mechcanisms and Projections are "receiver-oriented":
     - this the reason for the extra arg in __init__ for State (owner_mechanism) and Projection (receiver)

### Value Compatibility Constraints and Equivalences:

     Constraints
         "x <: y [<module.method>]" indicates x constrains y (y must be (compatiable with) x value or type),
                  implemented in module.method

         Main.iscompatible() is used to test for compatiblity

     1) Mechanism <: States
             a) self <: State.owner
                 [Mechanism.instantiate_state]
             b) self.inputState.value (InputState value) <: self.variable (executeMethod variable)
                 [Mechanism. instantiate_attributes_before_execute_method /
                 instantiate_input_states; InputState.validate_variable]
             c) self.paramsCurrent[param] <: ParameterState.value
                 [Mechanism. instantiate_attributes_before_execute_method  /
                  instantiate_execute_method_parameter_states]
             d) output of self.executeMethod <: self.outputState.value (OutputState value)
                 [Mechanism. instantiate_attributes_after_execute_method/instantiate_output_states;
                  OutputState.validate_variable]

     2) States value <: execute method
             Note: execute method simply updates value, so variable, output and value should all be compatible
             a) self.value <: self.variable (executeMethod variable)
                 [InputState.validate_variable]
             b) if number of mechanism.inputStates > 1:
                 number of mechanism.inputStates == length of self.variable
                 [MechainsmState.instantiate_mechanism_state_list]
             c) if number of mechanism.outputStates > 1:
                 number of mechanism.outstates == length of self.value
                 [MechainsmState.instantiate_mechanism_state_list]

     3) States : Projections:
             Note: any incompatibilities between projection output and receiver value raises an
             exception that must be corrected by the user (since can't force a modification in
             projection's execute method)
             a) State <: projections.receiver;
                 [Process.instantiate_configuration, State.instantiate_projection,
                  Projection.validate_states, ControlSignal.assign_states, Mapping.assign_states]
            b) self.sender.value : self.variable (executeMethod variable)
                [Projection.instantiate_attributes_before_execute_method / instantiate_sender]
            c) self.receiver.value = self.value
                [State.instantiate_projections_to_state, Projection.instantiate_execute_method]

     Equivalences (implied from above constraints):
         == equal values
         ~ compatible values or types (depends on constraint);  values may not be equal
     a) State execute method variable ~ output ~ State value
          note: State execute methods serve as update functions,
                so input, output, and value should all be the same format;
                however, they may not be equivalent in value, depending upon the update states of the mechanism
     b) Mechanism execute method variable == InputState value
     c) InputState value ~ InputState execute method variable
     d) OutputState value == OutputState variable
     e) ParameterState value ~ ParameterState execute method variable

### Parameters:

     paramClassDefaults:

         - Dictionary used to provide defaults for params of Function class and all of its subclasses
         - Subclasses should inherit super's paramClassDefaults, and their own
         - Entries added by one subclass should subclass-specific (i.e., not represented in sibling classes)
         - Subclasses should implement their copy of paramClassDefaults as follows:

          class SuperClass:
             paramClassDefaults = {<Parent’s defaults>}

         class SubClass(SuperClass):
             paramClassDefaults = SuperClass.paramClassDefaults.copy()
             paramClassDefaults.update({<SubClass additions>})

     #    - If a class requires a param to be implemented, it should enforce this in validate_params
         - If a class requires a param to be implemented, it should also include

     requiredParamClassDefaultTypes:

         - Dictionary used to specify params that are required for a given class and all subclasses
             and for which there is no default value to assign to paramClassDefaults
         - An exception is generated if a class fails to comply

         class SubClass(SuperClass):
             requiredParamClassDefaultTypes = SuperClass.requiredParamClassDefaultTypes.copy()
             requiredParamClassDefaultTypes.update({<required entries>})

     Parameter specification:
         - All Function objects have three sets of parameter values that determine how their execute method operates:
              + defaults defined for all parameters of the object's class, stored in paramClassDefaults (see above)
              + instance-specific values, stored in paramInstanceDefaults (if specified override paramClassDefaults)
              + current parameter values, stored in paramsCurrent, that are in effect for the current call to the object
         - Parameters are always specified as entries in a dict, with a:
              + key that identifies the parameter set
              + value that itself is a dict, the entries of which have a:
                  key that identifies the param
                  value that specifies the value to assign to the parmeter
              + example: kwInputStateParams:{<param_name>:value, <param_name>:value...}
         - Parameters can be specified:
             + on instantiation, in a dict passed as the params arg of the instantiation call:
                 the value(s) of the param(s) specified will be assigned to paramInstanceDefaults
                 they will override the value(s) in paramClassDefaults in all calls/references to the object
             + at runtime, in a dict passed as the params arg of the call to the object's execute method:
                 they will override the value(s) in paramInstanceDefaults ONLY FOR THE CURRENT CALL to the object
                 the value(s) in paramInstanceDefaults will be preserved, and used in subsequent calls
             + at runtime, in a dict passed as the second item of a (mechanism, params) in a configuration list:
                 they will override the value(s) in paramInstanceDefaults ONLY FOR THE CURRENT CALL to the object
                 the value(s) in paramInstanceDefaults will be preserved, and used in subsequent calls
                 note: this can only be used for State and ExecuteMethod params
         - As noted above, all params determine the operation of the object's execute method;
             + these are specified in a set identified by the keyword kwExecuteMethodParams
             + this can be included as the entry of the dict:
                 - in the params arg of a call to instantiate the object
                 - in the params arg of a call to execute the object's method
                 - in a (mechanism, params) tuple of a configuration list
                     in this case, the kwExecuteMethodParams entry must be contained in a dict that specifies the type of
                     object for which the params should be used;  this can be one of the following:
                         kwInputStateParams:  will be used for the execute method of the mechanism's inputState(s)
                         kwOutputStateParams:  will be used for the execute method of the mechanism's outputState(s)
                         kwParameterStateParams: will be used for the parameters of the mechanism's execute method
                     kwExecuteMethodParams can also be specified for projections to any of the states above, by including
                         kwExecuteMethodParams as an entry in one of the following dicts, that itself must be included in
                         one of the kwMechanism<state_type>Params dicts listed above:
                             kwProjectionParams: will apply for all projections to the specified state_type
                             kwMappingParams: will apply only to Mapping projections for the specified state_type
                             kwControlSignalParams: will apply only to ControlSignal projections for the specified state_type
                             <projection_name>: will apply only to projections with the specified name for the state_type

### Instantiation Sequence:

     Note: methods not implemented by subclass are shown in brackets (to see place in sequence)

     A) Function:
         1) Assign name
         2) Assign prefs
         3) Assign log
         4) Enforce implementation of variableClassDefault
         5) Enforce implementation of paramClassDefaults specified in requiredParamClassDefaultTypes
         5) assign_defaults
             a) validate_variable
                 - get value from ParamValueProjection tuple
                 - resolve function object or reference to current value
                 - insure variable is compatible with variableClassDefault (if variableClassDefault_locked == True)
                 - assign self.variable = variable
             b) assign missing params (if assign_missing == True)
             c) validate_params
                 - checks that each param is listed in paramClassDefaults
                 - checks that value is compatible with one in paramClassDefauts
         7) Set self.variable = variableInstanceDefault
         8) Set self.paramsCurrent = paramInstanceDefaults
         9) validate_execute_method
             - checks for valid method reference in paramsCurrent, paramInstanceDefaults, paramClassDefaults, and
                 finally self.execute;  if none present or valid, an exception is raised
         10) instantiate_attributes_before_execute_method: stub for subclasses
         11) instantiate_execute_method
             - instantiate params[kwExecuteMethod] if present and assign to self.executeMethod
             - else, instantiate self.execute; if it is not implemented, raise exception
             - call execute method to determine its output and type and assign to self.value
         12) instantiate_attributes_after_execute_method: stub for subclasses

     B) Process:
         1) Assign name
         2) Register category
         3) Assign prefs
         4) Assign log
         5) super.__init__:
             a) instantiate_attributes_after_execute_method
                 i) instantiate_configuration:
                     kwConfiguration:  must be a list of mechanism (object, class, or specification dict)
                 ii) super.instantiate_execute_method
         6) Set up log

     C) Mechanism:
         1) Validate that call is from subclass
         2) Assign name
         3) Register category
         4) Assign prefs
         5) Assign log
         6) super.__init__:
             a) validate_variable (for execute method)
                 insure that it is a value, consistent with variableClassDefault if variableClassDefault_locked is set
             b) validate_params:
                 kwTimeScale: must be TimeScale
                 kwInputStates;  must be a list or ordered dict, each item/entry of which is a:
                     InputState or Projection object or class ref, specification dict for one,
                     ParamValueProjection, or numberic value(s)
                 kwExecuteMethodParams; must be a dict, each entry of which must be a:
                     ParameterState or Projection object or class, specification dict for one,
                     ParamProjection tuple, or a value compatible with paramInstanceDefaults
                 kwOutputStates; must be a dict, each entry of which must be a:
                     InputState object or class, specification dict for one, or numeric value(s)
             [super: validate_execute_method]
             c) instantiate_attributes_before_execute_method
                 i) instantiate_inputStates
                     - inputState.value must be compatible with mechanism's variable
                     - State.instantiate_mechanism_states_list:
                         - assigns self.inputState (first/only state) and self.inputStates (OrderedDict of states)
                         - if number of inputStates > 1, must equal length of mechanism's variable
                             each state is assigned to an item of the mechanism's variable
                             if there is only one state, it is assigned to the full variable
                 ii) instantiate_execute_method_parameter_states
                     - assigns parameter state for each param in kwExecuteMethodParams
             [super: instantiate_execute_method]
             d) instantiate_attributes_after_execute_method
                 i) instantiate_outputStates - implement using kwOutputStates
                     - outputState.value must be compatible with output of mechanism's execute method
                     - State.instantiate_mechanism_states_list:
                         - assigns self.outputState (first/only state) and self.outputStates (OrderedDict of states)
                         - if number of outputStates > 1, must equal length of output of mechanism's execute method
                             each state is assigned an item of the output of the mechanism's execute method
                             if there is only one state, full output of mechanism's execute method is assigned to it
         7) Enforce class methods

     D) State:
         1) Validate that call is from subclass
         2) Assign name
         3) Register category
         4) Assign prefs
         5) Assign log
         6) Assign owner
         7) super.__init__:
             a) validate_variable:
                 insures that it is a number of list or tuple of numbers
                 assigns self.value to self.variable
             b) validate_params:
                 kwStateProjections:
                     must be a Projection object or class, or specification dict for one
                     specification dict must have the following entries::
                         kwProjectionType:<Projection class>
                         kwProjectionParams:<dict> - params for kwProjectionType
             c) instantiate_execute_method:
                 insures that output of execute method is compatible with state's value
         8) instantiate_projections_to_state:
             - each must be a Projection class or object or a specification dict for one
             - insures output of projection execute method is compatible with state.value
             - insures receiver for each projection is state
             - if spec is not valid, default is created of type determined by paramsCurrent[kwProjectionType]
             - adds each to state.receivesFromProjections
         9) Assign observers

     E) Projection:
         1) Validate subclass
         2) Assign name
         3) Register category
         4) Assign prefs
         5) Assign log
         6) Assign self.sender to sender arg
         7) Assign self.receiver to receiver arg
         8) super.__init__:
             [super: validate_variable]
             a) validate_params:
                 - kwProjectionSender and/or sender arg:
                     must be Mechanism or State
                 - gives precedence to kwProjectionSender, then sender arg, then default
             [super: validate_execute_method]
             b) instantiate_attributes_before_execute_method:
                 - calls instantiate_sender and instantiate_receiver (which both must be done before validate_execute_method)
                 i) instantiate_sender:
                     insures that projection's variable is compabitible with the output of the sender's execute method
                     if it is not, reassigns self.variable
                 ii) instantiate_receiver:
                     assigns (reference to) receiver's inputState to projection's receiver attribute
             c) instantiate_execute_method:
                 insures that output of projection's execute method is compatible with receiver's value
                 (it if it is a number of len=1, it tries modifying the output of execute method to match receiver)
                 checks if kwExecuteMethod is specified, then if self.execute implemented; raises exception if neither
             [super: instantiate_attributes_after_execute_method]
             
          E.1) LearningSignal:  
             1) Assign name
             2) super.__init__:
                 a) Assign self.sender to sender arg
                 b) Assign self.receiver to receiver arg
                 [super: validate_variable]
                 c) validate_params:
                     super():
                         - assign self.sender to sender arg or params[kwProjectionSender]
                         - gives precedence to kwProjectionSender, then sender arg, then paramClassDefaults
                         - validate that self.sender is Mechanism or State
                     LearningSignal:
                         - validate that self.sender is OutputState of MonitoringMechanism or ProcessingMechanism
                             or MonitoringMechanism class ref (assigned by paramClassDefaults)

                         ** DOCUMENT ??? GET kwParameterStates OR SET TO None?? 


                 [super: validate_execute_method]
                 d) instantiate_attributes_before_execute_method:
                     - calls instantiate_receiver and instantiate_sender (which both must be done before validate_execute_method)
                         * instantiate_receiver must be called before instantiate_sender since the latter requires access to
                             self.receiver to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
                     i) instantiate_receiver:
                         * doesn't call super() since that assumes self.receiver.owner is a Mechanism and calls add_projection_to_mechanism

        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, it is assigned to parameterStates[kwWeightMatrix]
            for the projection;  if that does not exist, it is instantiated and assigned as the receiver
        If specified as a ParameterState, validate that it is parameterStates[kwWeightMatrix]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        
        Note:
        * This must be called before instantiate_sender since that requires access to self.receiver
            to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
        * Doesn't call super().instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls add_projection_to_mechanism

                     ii) instantiate_sender:

                         insures that projection's variable is compabitible with the output of the sender's execute method
                         if it is not, reassigns self.variable
                         
        """Assign self.variable to MonitoringMechanism output or self.receiver.receiverErrorSignals 
        
        Call this after instantiate_receiver, as the latter may be needed to identify the MonitoringMechanism
        
        If sender arg or kwProjectionSender was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
            - assign its outputState.value as self.variable
        If sender was not specified (remains MonitoringMechanism_Base as specified in paramClassDefaults):
           if the owner of the Mapping projection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
               - assign its outputState.value as self.variable
           otherwise, if self.receiver.owner has an receiverError attribute, as that as self.variable
               (error signal for hidden units by BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception
                         
                         

                 e) instantiate_execute_method:
                     insures that output of projection's execute method is compatible with receiver's value
                     (it if it is a number of len=1, it tries modifying the output of execute method to match receiver)
                     checks if kwExecuteMethod is specified, then if self.execute implemented; raises exception if neither
                 [super: instantiate_attributes_after_execute_method]


### Execution Sequence:

     - Process.execute calls mechanism.update for each mechanism in its configuration in sequence
         - input specified as arg in execution of Process is provided as input to the first mechanism in configuration
         - output of last mechanism in configuration is assigned as Process.ouputState.value
         - DefaultController is executed before execution of each mechanism in the configuration
         - notes:
             * the same mechanism can be listed more than once in a configuration, inducing recurrent processing
             * if it is the first mechanism, it will receive its input from the Process only once (first execution)
     - Mechanism.update_states_and_execute:
         [TBI: calls each of the execute methods in its executionSequence (see Mechanism.execute):
         - calls self.inputState.update() for each entry in self.inputStates, which:
             + executes every self.inputState.receivesFromProjections.[<Projection>.execute()...]
                 note:  for the first mechanism in the configuration, this includes a projection with Process input
             + aggregates them using self.inputState.params[kwExecuteMethod]()
             + applies any runtime kwMechansimInputStateParams specified with mechanism in a tuple in the configuration
             + stores result in self.inputState.value
         - calls self.update_parameter_states, which calls every self.params[<ParameterState>].execute(),
             each of which:
             + executes self.params[<ParameterState>].receivesFromProjections.[<Projection>.execute()...]
                 (usually this absent, or is a single ControlSignal projection from DefaultController)
                 with any runtime kwMechansimParameterStateParams specified with mechanism in tupel in configuration
             + aggregates results using self.params[<ParameterState>].params[kwExecuteMethod]()
             + applies the result to self.params[<ParameterState>].baseValue
                 using self.params[<ParameterState>].paramsCurrent[kwParamModulationOperation] or runtime spec
         - calls subclass' self.update, which:
             + uses for each item of its variable the value of the corresponding state in mechanism's self.inputStates
             + uses self.params[<ParameterState>].value for each corresponding param of subclass' execute method
             + calls mechanism.execute method that carries out mechanism-specific computations
             + assigns each item of its output as the value of the corresponding state in mechanisms's self.outputStates
         - [TBI: calls self.outputState.execute() (output gating) to update self.outputState.value]

### Preferences:

     DOCUMENT: ADD DETAILS TO PREF DESCRIPTIONS BELOW
         - PreferenceSets:
             Each object has a prefs attribute that is assigned a PreferenceSet object specifying its preferences
             Preference objects have a set of preference attributes, one for each preference
             In addition to objects, every class in the Function hierarchy is assigned a:
             + PreferenceLevel: used to specify preferences for objects at that class level and below
             + PreferenceSet: preference settings for the corresponding level of specification (see PreferenceLevels below)

     - Standard preferences:
         Function objects and Mechanism subclass objects have the following preferences:
         Format:  pref_name (type): [Class]
         * note: "Class" refers to the class (and all subclasses) for which the preference is defined

         + verbose_pref (bool):  [Function]
             determines whether non-execute-related actions (e.g., initialization) and non-fatal warnings are reported

         + paramValidation_pref (bool):  [Function]
             determines whether the parameters of an object's execute method are validated prior to execution

         + reportOutput_pref (bool):  [Function]
             determines whether output of execute-related actions is reported to the console (see Process and Mechanism)

         + log_pref (LogPreferences): [Function]
             determines whether activity of the object is recorded in its log (see Logging)

         + executeMethodRuntimeParams_pref (ModulationOperation): [Mechanism]
             determines whether and, if so, how parameters passed to a mechanism at runtime influence its execution

     - PreferenceEntry:
         Each attribute of a PreferenceSet is a PreferenceEntry(setting, level) tuple:
         + setting (value):
             specifies the value of the preference, which must be of the type noted above
         + level (PreferenceLevel):
             specifies the level that will be used to determine that setting
             specifying a given level causes the value assigned at that level to be returned
             when a request is made for the value of the preference for that PreferenceSet

     - PreferenceLevels:
         There are four PreferenceLevels defined for the Function hierarchy:
         + System:  reserved for the Function class
         + Category: primary function subclasses (e.g., Process, Mechanism, State, Projection, Utility)
         + Type: Category subclasses (e.g., Mapping and ControlSignal subclasses of Projection, Utility subclasses)
         + Instance: an instance of an object of any class

     - Setting preferences:
         + Preferences settings can be assigned individually or in a PreferenceSet
             when instantiating or executing an object using the prefs arg, which can be:
             - a PreferenceSet, or
             - a specification dict with entries for each of the preferences to be set; for each entry the:
                 key must be a keyPath for a preference attribute
                     (kpVerbose, kpParamValidation, kpReportOutput, kpLog, kpExecuteMethodRuntimeParams)
                 value must be one of the following:
                     a PreferenceEntry(setting, level) tuple
                     a value that is valid for the setting of the corresponding attribute
                     a PreferenceLevel specification
         + a PreferenceSet can also be assigned directly to the preferences attirbute of an object or a class:
             <Object>.prefs = <PreferenceSet>
             <Class>.classPreferences = <PreferenceSet>
                Note:  if an assignment is made to a class, the class must be provided as the owner arg
                       in the call to instantiate the class (e.g.: my_pref_set = PreferenceSet(... owner=class... );
                       otherwise, an error will occur whenever the settings for the PreferenceSet are accessed
                       This is not required for objects; the owner of an object is determined automatically on assignment

         [TBI: + when specifying a configuration, in a (mechanism, params) tuple;  params must have:]

     - Show preferences:
         + Preferences for an object or class can be displayed by using inspect() method of a PreferenceSet:
             <object>.prefs.inspect(type) or <class>.classPreferences.inspect(type);
             both the base and current values of the setting are shown
             Note: these can be different if the PreferenceLevel is set to a value other than:
                INSTANCE for an object
                <class>.classLevel for a class

### Defaults:

     - System-wide:
         #Identifier (kwXXX):           # Class:                                 #Object:
         [TBI: SystemDefaultSender                                               ProcessDefaultInput]
         [TBI: SystemDefaultReceiver                                             ProcessDefaultOutput]
         kwDefaultProcessingMechanism       DefaultProcessingMechanism_Base              DefaultProcessingMechanism (in __init__.py)
         kwProcessDefaultMechanism      defaultMechanism (in Mechanism_Base)     Mechanism_Base.defaultMechanism
         kwSystemDefaultController      DefaultControlMechanism            DefaultController(in __init__.py)

     - Process:
         Single Default Mechanism (DDM)

     - Mechanism:
         DDM:
             InputState:
                 projections:
                     Mapping
                         sender: SystemDefaultSender
             OutputState:
                 [TBI: sender for projection to SystemDefaultReceiver]

     - State:
         ParameterState
             Projection:
                 ControlSignal
                     sender:  DefaultController)

### Key Value Observing (KVO):

     Observed object must implement the following:

     - a dictionary (in its __init__ method) of observers with an entry for each attribute to be observed:
         key should be string identifying attribute name
         value should be empty list
         TEMPLATE:
             self.observers = {<kpAttribute>: []}  # NOTE: entry should be empty here

     - a method that allows other objects to register to observe the observable attributes:
         TEMPLATE:
             def add_observer_for_keypath(self, object, keypath):
                 self.observers[keypath].append(object)

     - a method that sets the value of each attribute to be observed with the following format
         TEMPLATE:
             def set_attribute(self, new_value):
                 old_value = self.<attribute>
                 self.<attribute> = new_value
                 if len(self.observers[<kpAttribute>]):
                     for observer in self.observers[<kpAttribute>]:
                         observer.observe_value_at_keypath(<kpAttribute>, old_value, new_value)

     Observing object must implement a method that receives notifications of changes in the observed objects:
         TEMPLATE
             def observe_value_at_keypath(keypath, old_value, new_value):
                 [specify actions to be taken for each attribute (keypath) observed]
