# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  Mechanism ***********************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Globals.Registry import register_category

MechanismRegistry = {}


class MonitoredOutputStatesOption(AutoNumber):
    PRIMARY_OUTPUT_STATES = ()
    ALL_OUTPUT_STATES = ()
    NUM_MONITOR_STATES_OPTIONS = ()


class MechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# Mechanism factory method:
def mechanism(mech_spec=NotImplemented, params=NotImplemented, context=NotImplemented):
# DOCUMENT:  UPDATE:
    """Return subclass specified by mech_spec or default mechanism

    If called with no arguments or first argument is NotImplemented, instantiates default subclass (currently DDM)
    If called with a name string:
        - if it is registered in the MechanismRegistry class dictionary as the name of a subclass, instantiates that class
        - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
    If a params dictionary is included, it is passed to the subclass
    If called with context=kwValidate, return True if specification would return a valid class object; otherwise False

    :param mech_spec: (Mechanism class, descriptor keyword, or specification dict)
    :param params: (dict)
    :param context: (str)
    :return: (Mechanism object or None)
    """

    # Called with descriptor keyword
    if mech_spec in MechanismRegistry:
        return MechanismRegistry[mech_spec].mechanismSubclass(params=params, context=context)

    # Called with a string that is not in the Registry, so return default type with the name specified by the string
    elif isinstance(mech_spec, str):
        return Mechanism_Base.defaultMechanism(name=mech_spec, params=params, context=context)

    # Called with a Mechanism type, so return instantiation of that type
    elif isclass(mech_spec) and issubclass(mech_spec, Mechanism):
        return mech_spec(params=params, context=context)

    # Called with Mechanism specification dict (with type and params as entries within it), so:
    #    - get mech_type from kwMechanismType entry in dict
    #    - pass all other entries as params
    elif isinstance(mech_spec, dict):
        # Get Mechanism type from kwMechanismType entry of specification dict
        try:
            mech_spec = mech_spec[kwMechanismType]
        # kwMechanismType config_entry is missing (or mis-specified), so use default (and warn if in VERBOSE mode)
        except (KeyError, NameError):
            if Mechanism.classPreferences.verbosePref:
                print("{0} entry missing from mechanisms dict specification ({1}); default ({2}) will be used".
                      format(kwMechanismType, mech_spec, Mechanism_Base.defaultMechanism))
            return Mechanism_Base.defaultMechanism(name=kwProcessDefaultMechanism, context=context)
        # Instantiate Mechanism using mech_spec dict as arguments
        else:
            return mech_spec(context=context, **mech_spec)

    # Called without a specification, so return default type
    elif mech_spec is NotImplemented:
        return Mechanism_Base.defaultMechanism(name=kwProcessDefaultMechanism, context=context)

    # Can't be anything else, so return empty
    else:
        return None


class Mechanism_Base(Mechanism):
# DOCUMENT: PHASE_SPEC;  (??CONSIDER ADDING kwPhaseSpec FOR DEFAULT VALUE)
    """Implement abstract class for Mechanism category of Function class (default type:  DDM)

    Description:
        Mechanisms are used as part of a configuration (together with projections) to execute a process
        A mechanism is associated with a name, a set of states, an update_and_execute method, and an execute method:
        - one or more inputStates:
            the value of each represents the aggregated input from its incoming mapping projections, and is used as
            the corresponding item in the variable of the mechanism's execute method
            * Note:
                by default, a Mechanism has only one inputState, assigned to <mechanism>.inputState;  however:
                if the specification of the Mechanism's variable is a list of arrays (lists), or
                if params[kwinputStates] is a list (of names) or a specification dict (of MechanismInputState specs),
                    <mechanism>.inputStates (note plural) is created and contains a dict of inputStates,
                    the first of which points to <mechanism>.inputState (note singular)
        - a set of parameters, each of which must be (or resolve to) a reference to a MechanismParameterState
            these determine the operation of the mechanism's execute method
        - one or more outputStates:
            the variable of each receives the corresponding item in the output of the mechanism's execute method
            the value of each is passed to corresponding mapping projections for which the mechanism is a sender
            Notes:
            * by default, a Mechanism has only one outputState, assigned to <mechanism>.outputState;  however:
                  if params[kwOutputStates] is a list (of names) or specification dict (of MechanismOuput State specs),
                  <mechanism>.outputStates (note plural) is created and contains a dict of outputStates,
                  the first of which points to <mechanism>.outputState (note singular)
            [TBI * each outputState maintains a list of projections for which it serves as the sender]

        - an update_states_and_execute method:
            coordinates updating of inputs, params and execution of mechanism's execute method (implemented by subclass)
        Constraints:
            - the number of inputStates must correspond to the length of the variable of the mechanism's execute method
            - the value of each inputState must be compatible with the corresponding item in the
                variable of the mechanism's execute method
            - the value of each parameterState must be compatible with the corresponding parameter of  the mechanism's
                 execute method
            - the number of outputStates must correspond to the length of the output of the mechanism's execute method,
                (self.value)
            - the value of each outputState must be compatible with the corresponding item of the self.value
                 (the output of the mechanism's execute method)
    Subclasses:
        The implemented subclasses are:
            - SystemDefaultMechanism_Base (used for SystemDefaultInputMechanism and SystemDefaultOutputMechanism)
            - DDM (default Mechanism)
            - SystemDefaultControlMechanism (used for SystemDefaultController)

    Instantiation:
        Mechanisms should NEVER be instantiated by a direct call to the class
        A Mechanism can be instantiated in one of several ways:
        - by calling the mechanism() module factory method, which instantiates the default mechanism (currently DDM)
            • the nth instance created will be named using the following format: functionType-n
        - by calling mechanism(name, params):
            • if name is the name of a mechanism class in the MechanismRegistry dictionary, it will be instantiated;
                otherwise, the default mechanism will be instantiated with that name
            • params (optional) must be a dictionary with parameter values relevant to the class invoked (see below)
        - by calling a subclass directly (e.g., DDM(name, params)); the name will be assigned to the instance
        - any parameters included in the initialization will be used as defaults for all calls to the mechanism
        - whenever a new subclass is instantiated (either through import or by one of the methods above),
            it is registered in the MechanismRegistry, which keeps a record of the subClass and instanceCount

    Initialization arguments:
        - variable:  establishes type of variable for the execute method, and initializes it (default: ??)
        - params (dict): (see validate_params and instantiate_mechanism_state below for implementation details)
            + kwMechanismInputState (value, list, dict):
                if param is absent:
                   a default MechanismInputState will be instantiated using variable of mechanism's execute method (EMV)
                    it will be placed as the single entry in an OrderedDict
                if param is a single value:
                    it will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is a list:
                    each item will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is an OrderedDict:
                    each entry will (if necessary) be instantiated as a MechanismInputState
                in each case, the result will be an OrderedDict of one or more entries:
                    the key for the entry will be the name of the inputState if provided, otherwise
                        kwMechanismInputStates-n will used (with n incremented for each entry)
                    the value of the inputState in each entry will be used as the corresponding value of the EMV
                    the dict will be assigned to both self.inputStates and paramsCurrent[kwMechanismInputState]
                    self.inputState will be pointed to self.inputStates[0] (the first entry of the dict)
                notes:
                    * if there is only one inputState, but the EMV has more than one item, it is assigned to the
                        the sole inputState, which is assumed to have a multi-item value
                    * if there is more than one inputState, number must match length of EMV, or an exception is raised
                specification of the param value, list item, or dict enrty value can be any of the following,
                    as long as it is compatible with the variable of the mechanism's execute method (EMV):
                    + MechanismInputState class: default will be instantiated using EMV as its value
                    + MechanismInputState object: its value must be compatible with EMV
                    + Projection subclass ref:
                        default MechanismInputState will be instantiated using EMV as its value
                        default projection (for MechanismInputState) will be instantiated using EMV as its variable
                            and assigned to MechanismInputState
                    + Projection object:
                        MechanismInputState will be instantiated using output of projection as its value;
                        this must be compatible with EMV
                    + specification dict:  MechanismInputState will be instantiated using EMV as its value;
                        must contain the following entries: (see Initialization arguments for MechanismState):
                            + kwExecuteMethod (method)
                            + kwExecuteMethodParams (dict)
                            + kwMechanismStateProjections (Projection, specifications dict, or list of either of these)
                    + ParamValueProjection:
                        value will be used as variable to instantiate a default MechanismInputState
                        projection will be assigned as projection to MechanismInputState
                    + value: will be used as variable to instantiate a default MechanismInputState
                * note: inputStates can also be added using <Mechanism>.instantiate_mechanism_state (see MechanismState)
            + kwExecuteMethod:(method):  method used to transform mechanism input to its output;
                this must be implemented by the subclass, or an exception will be raised
                each item in the variable of this method must be compatible with the corresponding MechanismInputState
                each item in the output of this method must be compatible  with the corresponding MechanismOutputState
                for any parameter of the method that has been assigned a MechanismParameterState,
                    the output of the parameter state's own execute method must be compatible with
                    the value of the parameter with the same name in paramsCurrent[kwExecuteMethodParams] (EMP)
            + kwExecuteMethodParams (dict):
                if param is absent, no parameterStates will be created
                if present, each entry will (if necessary) be instantiated as a MechanismParameterState,
                    and the resulting dict will be placed in <mechanism>.executeMethodParameterStates
                the value of each entry can be any of those below, as long as it resolves to a value that is
                    compatible with param of the same name in <mechanism>.paramsCurrent[kwExecuteMethodParams] (EMP)
                    + MechanismParameterState class ref: default will be instantiated using param with same name in EMP
                    + MechanismParameterState object: its value must be compatible with param of same name in EMP
                    + Projection subclass ref:
                        default MechanismParameterState will be instantiated using EMP
                        default projection (for MechanismParameterState) will be instantiated using EMP
                            and assigned to MechanismParameterState
                    + Projection object:
                        MechanismParameterState will be instantiated using output of projection as its value;
                        this must be compatible with EMP
                    + specification dict:  MechanismParameterState will be instantiated using EMP as its value;
                        must contain the following entries: (see Instantiation arguments for MechanismParameterState):
                            + kwExecuteMethod (method)
                            + kwExecuteMethodParams (dict)
                            + kwMechanismStateProjections (Projection, specifications dict, or list of either of these)
                    + ParamValueProjection tuple:
                        value will be used as variable to instantiate a default MechanismParameterState
                        projection will be assigned as projection to MechanismParameterState
                    + 2-item tuple [convenience notation;  should use ParamValueProjection for clarity]:
                        first item will be used as variable to instantiate a default MechanismParameterState
                        second item will be assigned as projection to MechanismParameterState
                    + value: will be used as variable to instantiate a default MechanismParameterState
            + kwMechanismOutputStates (value, list, dict):
                if param is absent:
                    a default MechanismOutputState will be instantiated using output of mechanism's execute method (EMO)
                    it will be placed as the single entry in an OrderedDict
                if param is a single value:
                    it will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is a list:
                    each item will (if necessary) be instantiated and placed in an OrderedDict
                if param is an OrderedDict:
                    each entry will (if necessary) be instantiated as a MechanismOutputState
                in each case, the result will be an OrderedDict of one or more entries:
                    the key for the entry will be the name of the outputState if provided, otherwise
                        kwMechanismOutputStates-n will used (with n incremented for each entry)
                    the value of the outputState in each entry will be assigned to the corresponding item of the EMO
                    the dict will be assigned to both self.outputStates and paramsCurrent[kwMechanismOutputStates]
                    self.outputState will be pointed to self.outputStates[0] (the first entry of the dict)
                notes:
                    * if there is only one outputState, but the EMV has more than one item, it is assigned to the
                        the sole outputState, which is assumed to have a multi-item value
                    * if there is more than one outputState, number must match length of EMO, or an exception is raised
                specification of the param value, list item, or dict entry value can be any of the following,
                    as long as it is compatible with (relevant item of) output of the mechanism's execute method (EMO):
                    + MechanismOutputState class: default outputState will be instantiated using EMO as its value
                    + MechanismOutputState object: its value must be compatible with EMO
                    + specification dict:  MechanismOutputState will be instantiated using EMO as its value;
                        must contain the following entries: (see Initialization arguments for MechanismState):
                            + kwExecuteMethod (method)
                            + kwExecuteMethodParams (dict)
                    + str:
                        will be used as name of a default outputState (and key for its entry in self.outputStates)
                        value must match value of the corresponding item of the mechanism's EMO
                    + value:
                        will be used a variable to instantiate a MechanismOutputState; value must be compatible with EMO
                * note: inputStates can also be added using <Mechanism>.instantiate_mechanism_state
            + kwMonitoredOutputStates (list): (default: PRIMARY_OUTPUT_STATES)
                specifies the outputStates of the mechanism to be monitored by SystemControlMechanism of the System(s)
                    to which the Mechanism belongs
                this specification overrides (for this Mechanism) any in the SystemControlMechanism or System params[]
                this is overridden if None is specified for kwMonitoredOutputStates in the outputState itself
                each item must be one of the following:
                    + MechanismOutputState (object)
                    + MechanismOutputState name (str)
                    + (Mechanism or MechanismOutputState specification, exponent, weight) (tuple):
                        + mechanism or outputState specification (Mechanism, MechanismOutputState, or str):
                            referenceto Mechanism or MechanismOutputState object or the name of one
                            if a Mechanism ref, exponent and weight will apply to all outputStates of that mechanism
                        + exponent (int):  will be used to exponentiate outState.value when computing EVC
                        + weight (int): will be used to multiplicative weight outState.value when computing EVC
                    + MonitoredOutputStatesOption (AutoNumber enum): (note: ignored if one of the above is specified)
                        + PRIMARY_OUTPUT_STATES:  monitor only the primary (first) outputState of the Mechanism
                        + ALL_OUTPUT_STATES:  monitor all of the outputStates of the Mechanism
                    + Mechanism (object): ignored (used for SystemController and System params)
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value 
             (see Description under PreferenceSet for details)
        - context (str): must be a reference to a subclass, or an exception will be raised

    MechanismRegistry:
        All Mechanisms are registered in MechanismRegistry, which maintains a dict for each subclass,
          a count for all instances of that type, and a dictionary of those instances

    Naming:
        Mechanisms can be named explicitly (using the name='<name>' argument).  If the argument is omitted,
        it will be assigned the subclass name with a hyphenated, indexed suffix ('subclass.name-n')

    Class attributes:
        + functionCategory = kwMechanismFunctionCategory
        + className = functionCategory
        + suffix = " <className>"
        + className (str): kwMechanismFunctionCategory
        + suffix (str): " <className>"
        + registry (dict): MechanismRegistry
        + classPreference (PreferenceSet): Mechanism_BasePreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault (list)
        + paramClassDefaults (dict):
            + kwMechanismTimeScale (TimeScale): TimeScale.TRIAL (timeScale at which mechanism executes)
            + [TBI: kwMechanismExecutionSequenceTemplate (list of MechanismStates):
                specifies order in which types of MechanismStates are executed;  used by self.execute]
        + paramNames (dict)
        + defaultMechanism (str): Currently kwDDM (class reference resolved in __init__.py)

    Class methods:
        • validate_variable(variable, context)
        • validate_params(request_set, target_set, context)
        • instantiate_mechanism_state_list(state_type,
                                           state_param_identifier,
                                           constraint_values,
                                           constraint_values_name,
                                           context)
            instantiates states of type specified from list in paramsCurrent specified by state_param_identifier;
            passes state_type and constraints to instantiate_mechanism_state for instantiating each individual state
        • instantiate_mechanism_state(state_type
                                      state_name,
                                      state_spec,
                                      constraint_values,
                                      constraint_values_name,
                                      constraint_index,
                                      context):
            instantiates state of type specified by state_type and state_spec, using constraints
        • instantiate_input_states(context)
        • instantiate_execute_method_parameter_states(context)
        • instantiate_output_states(context)
        • update_states_and_execute(time_scale, params, context):
            updates input, param values, executes <subclass>.function, returns outputState.value
        • terminate_execute(self, context=NotImplemented): terminates execution of mechanism (for TimeScale = time_step)
        • adjust(params, context)
            modifies specified mechanism params (by calling Function.assign_defaults)
            returns output
            # parses, validates and assigns validated control signal (called by __init__() and adjust() class methods)

    Instance attributes (implemented in __init__():
        + variable (value) - used as input to mechanism's execute method
        + paramsCurrent (dict) - current value of all params for instance (created and validated in Functions init)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramNames (list) - list of keys for the params in paramInstanceDefaults
        + inputState (MechanismInputState) - default MechanismInput object for mechanism
        + inputStates (dict) - created if params[kwMechanismInputState] specifies  more than one MechanismInputState
        + inputValue (value, list or ndarray) - value, list or array of values, one for the value of each inputState
        + receivesProcessInput (bool) - flags if Mechanism (as first in Configuration) receives Process input projection
        + executeMethodParameterStates (dict) - created if params[kwExecuteMethodParams] specifies any parameters
        + outputState (MechanismOutputState) - default MechanismOutputState for mechanism
        + outputStates (dict) - created if params[kwMechanismOutputStates] specifies more than one MechanismOutputState
        + value (value, list, or ndarray) - output of the Mechanism's execute method;
            Note: currently each item of self.value corresponds to value of corresponding outputState in outputStates
        + phaseSpec (int or float) - time_step(s) on which Mechanism.update() is called (see Process for specification)
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying Mechanism_BasePreferenceSet

    Instance methods:
        The following method MUST be overridden by an implementation in the subclass:
        • execute:
            - called by update_states_and_execute()
            - must be implemented by Mechanism subclass, or an exception is raised
        [TBI: • terminate(context) -
            terminates the process
            returns output
    """

    #region CLASS ATTRIBUTES
    functionCategory = kwMechanismFunctionCategory
    className = functionCategory
    suffix = " " + className

    registry = MechanismRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # Any preferences specified below will override those specified in CategoryDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to CATEGORY automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'MechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # IMPLEMENTATION NOTE: move this to a preference
    defaultMechanism = kwDDM

    variableClassDefault = [0.0]
    # Note:  the following enforce encoding as 2D np.ndarrays,
    #        to accomodate multiple states:  one 1D np.ndarray per state
    variableEncodingDim = 2
    valueEncodingDim = 2

    # Category specific defaults:
    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwMechanismTimeScale: TimeScale.TRIAL,
        # TBI - kwMechanismExecutionSequenceTemplate: [
        #     Functions.MechanismStates.MechanismInputState.MechanismInputState,
        #     Functions.MechanismStates.MechanismParameterState.MechanismParameterState,
        #     Functions.MechanismStates.MechanismOutputState.MechanismOutputState]
        })

    # def __new__(cls, *args, **kwargs):
    # def __new__(cls, name=NotImplemented, params=NotImplemented, context=NotImplemented):
    #endregion

    def __init__(self,
                 variable=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign name, category-level preferences, register mechanism, and enforce category methods

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

        Initialization arguments:
            - input_template (value, MechanismInputState or specification dict for one):
                  if value, it will be used as variable (template of self.inputState.value)
                  if MechanismState or specification dict, it's value attribute will be used
            - params (dict): dictionary with entries for each param of the mechanism subclass;
                the key for each entry should be the name of the param (used to name its associated projections)
                the value for each entry MUST be one of the following (see Parameters above for details):
                    - MechanismParameterState object
                    - dict: MechanismState specifications (see MechanismState)
                    - projection: Projection object, Projection specifications dict, or list of either)
                    - tuple: (value, projectionType)
                    - value: list of numbers (no projections will be assigned)
            - name (str):
                if provided, will set self.name of the instance to the name specified
                if absent, will assign the subclass name with a hyphenated, indexed suffix ('subclass.name-n')
            - context (str) - must be name of subclass;  otherwise raises an exception for direct call

        NOTES:
        * Since Mechanism is a subclass of Function, it calls super.__init__
            to validate variable_default and param_defaults, and assign params to paramInstanceDefaults;
            it uses kwInputState as the variable_default
        * registers mechanism with MechanismRegistry

        :param input_template: (value or MechanismDict)
        :param params: (dict)
        :param name: (str)
        :param prefs: (dict)
        :param context: (str)
        :return: None
        """

        # if not isinstance(context, self):
        if not isinstance(context, type(self)) and not kwValidate in context:
            raise MechanismError("Direct call to abstract class Mechanism() is not allowed; "
                                 "use mechanism() or one of the following subclasses: {0}".
                                 format(", ".join("{!s}".format(key) for (key) in MechanismRegistry.keys())))

# IMPLEMENT **args (PER MechanismState)

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        if not context is kwValidate:
            register_category(self, Mechanism_Base, MechanismRegistry, context=context)

        if context is NotImplemented or isinstance(context, object) or inspect.isclass(context):
            context = kwInit + self.name + kwSeparatorBar + self.__class__.__name__
        else:
            context = context + kwSeparatorBar + kwInit + self.name

        super(Mechanism_Base, self).__init__(variable_default=variable,
                                             param_defaults=params,
                                             prefs=prefs,
                                             name=self.name,
                                             context=context)

        # FUNCTIONS:

# IMPLEMENTATION NOTE:  REPLACE THIS WITH ABC (ABSTRACT CLASS)
        # Assign class functions
        self.classMethods = {
            kwMechanismExecuteFunction: self.execute,
            kwMechanismAdjustFunction: self.adjust_function,
            kwMechanismTerminateFunction: self.terminate_execute
            # kwMechanismAccuracyFunction: self.accuracy_function
        }
        self.classMethodNames = self.classMethods.keys()

        #  Validate class methods:
        #    make sure all required ones have been implemented in (i.e., overridden by) subclass
        for name, method in self.classMethods.items():
            try:
                method
            except (AttributeError):
                raise MechanismError("{0} is not implemented in mechanism class {1}".
                                     format(name, self.name))

        self.value = None
        self.receivesProcessInput = False
        self.phaseSpec = None

    def validate_variable(self, variable, context=NotImplemented):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        # VARIABLE SPECIFICATION:                                        ENCODING:
        # Simple value variable:                                         0 -> [array([0])]
        # Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        # Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        :param variable:
        :param context:
        :return:
        """

        super(Mechanism_Base, self).validate_variable(variable, context)

        # Force Mechanism variable specification to be a 2D array (to accomodate multiple input states - see above):
        # Note: instantiate_input_states (below) will parse into 1D arrays, one for each input state
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """validate TimeScale, inputState(s), execute method param(s) and outputState(s)

        Call super (Function.validate_params()
        Go through target_set params (populated by Function.validate_params) and validate values for:
            + kwTimeScale:  <TimeScale>
            + kwMechanismInputState:
                <MechanismsInputState or Projection object or class,
                specification dict for one, ParamValueProjection tuple, or numeric value(s)>;
                if it is missing or not one of the above types, it is set to self.variable
            + kwExecuteMethodParams:  <dict>, every entry of which must be one of the following:
                MechanismParameterState or Projection object or class, specification dict for one,
                ParamValueProjection tuple, or numeric value(s);
                if invalid, default (from paramInstanceDefaults or paramClassDefaults) is assigned
            + kwMechanismOutputStates:
                <MechanismsOutputState object or class, specification dict, or numeric value(s);
                if it is missing or not one of the above types, it is set to None here;
                    and then to default value of self.value (output of execute method) in instantiate_outputState
                    (since execute method must be instantiated before self.value is known)
                if kwMechanismOutputStates is a list or OrderedDict, it is passed along (to instantiate_outputStates)
                if it is a MechanismOutputState class ref, object or specification dict, it is placed in a list

        TBI - Generalize to go through all params, reading from each its type (from a registry),
                                   and calling on corresponding subclass to get default values (if param not found)
                                   (as kwProjectionType and kwProjectionSender are currently handled)

        :param request_set: (dict)
        :param target_set: (dict)
        :param context: (str)
        :return:
        """

        # Perform first-pass validation in Function.__init__():
        # - returns full set of params based on subclass paramClassDefaults
        super(Mechanism, self).validate_params(request_set,target_set,context)

        params = target_set

        #region VALIDATE TIME SCALE
        try:
            param_value = params[kwTimeScale]
        except KeyError:
            self.timeScale = timeScaleSystemDefault
        else:
            if isinstance(param_value, TimeScale):
                self.timeScale = params[kwTimeScale]
            else:
                if self.prefs.verbosePref:
                    print("Value for {0} ({1}) param of {2} must be of type {3};  default will be used: {4}".
                          format(kwTimeScale, param_value, self.name, type(TimeScale), timeScaleSystemDefault))
        #endregion

        #region VALIDATE INPUT STATE(S)

        # MODIFIED 6/10/16
        # FIX: SHOULD CHECK LENGTH OF kwInputStates PARAM (LIST OF NAMES OR SPECIFICATION DICT) AGAINST LENGTH OF self.variable 2D ARRAY
        # FIX:                AND COMPARE variable SPECS, IF PROVIDED, WITH CORRESPONDING ELEMENTS OF self.variable 2D ARRAY
        try:
            param_value = params[kwMechanismInputStates]
            
        except KeyError:
            # kwMechanismInputStates not specified:
            # - set to None, so that it is set to default (self.variable) in instantiate_inputState
            # - if in VERBOSE mode, warn in instantiate_inputState, where default value is known
            params[kwMechanismInputStates] = None

        else:
            # kwMechanismInputStates is specified, so validate:
            # If it is a single item or a non-OrderedDict, place in a list (for use here and in instantiate_inputState)
            if not isinstance(param_value, (list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            # Note:
            # * number of inputStates is validated against length of the owner mechanism's execute method variable (EMV)
            #     in instantiate_inputState, where an inputState is assigned to each item (value) of the EMV
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from Functions.MechanismStates.MechanismInputState import MechanismInputState
                # If not valid...
                if not ((isclass(item) and (issubclass(item, MechanismInputState) or # MechanismInputState class ref
                                                issubclass(item, Projection))) or    # Project class ref
                            isinstance(item, MechanismInputState) or      # MechanismInputState object
                            isinstance(item, dict) or                     # MechanismInputState specification dict
                            isinstance(item, ParamValueProjection) or     # ParamValueProjection tuple
                            isinstance(item, str) or                      # Name (to be used as key in inputStates dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):   # value
                    # set to None, so it is set to default (self.variable) in instantiate_inputState
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " MechanismInputState, specification dict or value, nor a list of dict of them; "
                              "variable ({4}) of execute method for {5} will be used"
                              " to create a default outputState for {3}".
                              format(i,
                                     kwMechanismInputStates,
                                     param_value,
                                     self.__class__.__name__,
                                     self.variable,
                                     self.execute.__self__.name))
                i += 1
            params[kwMechanismInputStates] = param_value
        #endregion

        #region VALIDATE EXECUTE METHOD PARAMS
        try:
            execute_method_param_specs = params[kwExecuteMethodParams]
        except KeyError:
            if self.prefs.verbosePref:
                print("No params specified for {0}".format(self.__class__.__name__))
        else:
            if not (isinstance(execute_method_param_specs, dict)):
                raise MechanismError("{0} in {1} must be a dict of param specifications".
                                     format(kwExecuteMethodParams, self.__class__.__name__))
            # Validate params
            from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
            for param_name, param_value in execute_method_param_specs.items():
                try:
                    default_value = self.paramInstanceDefaults[kwExecuteMethodParams][param_name]
                except KeyError:
                    raise MechanismError("{0} not recognized as a param of execute method for {1}".
                                         format(param_name, self.__class__.__name__))
                if not ((isclass(param_value) and
                             (issubclass(param_value, MechanismParameterState) or
                                  issubclass(param_value, Projection))) or
                        isinstance(param_value, MechanismParameterState) or
                        isinstance(param_value, Projection) or
                        isinstance(param_value, dict) or
                        isinstance(param_value, ParamValueProjection) or
                        iscompatible(param_value, default_value)):
                    params[kwExecuteMethodParams][param_name] = default_value
                    if self.prefs.verbosePref:
                        print("{0} param ({1}) for execute method {2} of {3} is not a MechanismParameterState, "
                              "projection, ParamValueProjection, or value; default value ({4}) will be used".
                              format(param_name,
                                     param_value,
                                     self.execute.__self__.functionName,
                                     self.__class__.__name__,
                                     default_value))
        #endregion
        # FIX: MAKE SURE OUTPUT OF EXECUTE FUNCTION / SELF.VALUE  IS 2D ARRAY, WITH LENGTH == NUM OUTPUT STATES

        #region VALIDATE OUTPUT STATE(S)

        # FIX: MAKE SURE # OF OUTPUTS == LENGTH OF OUTPUT OF EXECUTE FUNCTION / SELF.VALUE
        try:
            param_value = params[kwMechanismOutputStates]

        except KeyError:
            # kwMechanismOutputStates not specified:
            # - set to None, so that it is set to default (self.value) in instantiate_outputState
            # Notes:
            # * if in VERBOSE mode, warning will be issued in instantiate_outputState, where default value is known
            # * number of outputStates is validated against length of owner mechanism's execute method output (EMO)
            #     in instantiate_outputState, where an outputState is assigned to each item (value) of the EMO
            params[kwMechanismOutputStates] = None

        else:
            # kwMechanismOutputStates is specified, so validate:
            # If it is a single item or a non-OrderedDict, place in a list (for use here and in instantiate_outputState)
            if not isinstance(param_value, (list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
                # If not valid...
                if not ((isclass(item) and issubclass(item, MechanismOutputState)) or # MechanismOutputState class ref
                            isinstance(item, MechanismOutputState) or   # MechanismOutputState object
                            isinstance(item, dict) or                   # MechanismOutputState specification dict
                            isinstance(item, str) or                    # Name (to be used as key in outputStates dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):  # value
                    # set to None, so it is set to default (self.value) in instantiate_outputState
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " MechanismOutputState, specification dict or value, nor a list of dict of them; "
                              "output ({4}) of execute method for {5} will be used"
                              " to create a default outputState for {3}".
                              format(i,
                                     kwMechanismOutputStates,
                                     param_value,
                                     self.__class__.__name__,
                                     self.value,
                                     self.execute.__self__.name))
                i += 1
            params[kwMechanismOutputStates] = param_value

        # MODIFIED 7/13/16 NEW: [MOVED FROM EVCMechanism]
        # FIX: MOVE THIS TO FUNCTION, OR ECHO IN SYSTEM
        #region VALIDATE MONITORED STATES (for use by SystemControlMechanism)
        # Note: this must be validated after kwMechanismOutputStates as it can reference entries in that param
        try:
            # It is a MonitoredOutputStatesOption specification
            if isinstance(target_set[kwMonitoredOutputStates], MonitoredOutputStatesOption):
                # Put in a list (standard format for processing by instantiate_monitored_output_states)
                target_set[kwMonitoredOutputStates] = [target_set[kwMonitoredOutputStates]]
            # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or MechanismStates
            else:
                # Validate each item of kwMonitoredOutputStates
                for item in target_set[kwMonitoredOutputStates]:
                    self.validate_monitored_state(item, context=context)
                # FIX: PRINT WARNING (IF VERBOSE) IF kwWeights or kwExponents IS SPECIFIED,
                # FIX:     INDICATING THAT IT WILL BE IGNORED;
                # FIX:     weights AND exponents ARE SPECIFIED IN TUPLES
                # FIX:     kwWeights and kwExponents ARE VALIDATED IN SystemContro.Mechanisminstantiate_monitored_output_states
                # # Validate kwWeights if it is specified
                # try:
                #     num_weights = len(target_set[kwExecuteMethodParams][kwWeights])
                # except KeyError:
                #     # kwWeights not specified, so ignore
                #     pass
                # else:
                #     # Insure that number of weights specified in kwWeights
                #     #    equals the number of states instantiated from kwMonitoredOutputStates
                #     num_monitored_states = len(target_set[kwMonitoredOutputStates])
                #     if not num_weights != num_monitored_states:
                #         raise MechanismError("Number of entries ({0}) in kwWeights of kwExecuteMethodParam for EVC "
                #                        "does not match the number of monitored states ({1})".
                #                        format(num_weights, num_monitored_states))
        except KeyError:
            pass
        #endregion
        # MODIFIED END

# FIX: MAKE THIS A CLASS METHOD OR MODULE FUNCTION
# FIX:     SO THAT IT CAN BE CALLED BY System TO VALIDATE IT'S kwMonitoredOutputStates param
    def validate_monitored_state(self, state_spec, context=NotImplemented):
        """Validate specification is a Mechanism or MechanismOutputState object or the name of one

        Called by both self.validate_params() and self.add_monitored_state() (in SystemControlMechanism)
        """
        state_spec_is_OK = False

        if isinstance(state_spec, MonitoredOutputStatesOption):
            state_spec_is_OK = True

        if isinstance(state_spec, tuple):
            if len(state_spec) != 3:
                raise MechanismError("Specification of tuple ({0}) in kwMonitoredOutputStates for {1} "
                                     "has {2} items;  it should be 3".
                                     format(state_spec, self.name, len(state_spec)))

            if not isinstance(state_spec[1], numbers.Number):
                raise MechanismError("Specification of the exponent ({0}) for kwMonitoredOutputStates of {1} "
                                     "must be a number".
                                     format(state_spec, self.name, state_spec[0]))

            if not isinstance(state_spec[2], numbers.Number):
                raise MechanismError("Specification of the weight ({0}) for kwMonitoredOutputStates of {1} "
                                     "must be a number".
                                     format(state_spec, self.name, state_spec[0]))

            # Set state_spec to the output_state item for validation below
            state_spec = state_spec[0]

        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        if isinstance(state_spec, (Mechanism, MechanismOutputState)):
            state_spec_is_OK = True

        if isinstance(state_spec, str):
            if state_spec in self.paramInstanceDefaults[kwMechanismOutputStates]:
                state_spec_is_OK = True
        try:
            self.outputStates[state_spec]
        except (KeyError, AttributeError):
            pass
        else:
            state_spec_is_OK = True

        if not state_spec_is_OK:
            raise MechanismError("Specification ({0}) in kwMonitoredOutputStates for {1} is not "
                                 "a Mechanism or MechanismOutputState object or the name of one".
                                 format(state_spec, self.name))
#endregion

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        self.instantiate_input_states(context=context)
        self.instantiate_execute_method_parameter_states(context=context)

    def instantiate_attributes_after_execute_method(self, context=NotImplemented):
        self.instantiate_output_states(context=context)

    def instantiate_input_states(self, context=NotImplemented):
        """Call instantiate_mechanism_state_list to instantiate orderedDict of inputState(s)

        Create OrderedDict of inputState(s) specified in paramsCurrent[kwMechanismInputStates]
        If kwMechanismInputStates is not specified, use self.variable to create a default input state
        When completed:
            - self.inputStates contains an OrderedDict of one or more inputStates
            - self.inputState contains first or only inputState in OrderedDict
            - paramsCurrent[kwMechanismOutputStates] contains the same OrderedDict (of one or more inputStates)
            - each inputState corresponds to an item in the variable of the mechanism's execute method (EMV)
            - if there is only one inputState, it is assigned the full value

        Note: instantiate_mechanism_state_list parses self.variable (2D np.array, passed in constraint_values)
              into individual 1D arrays, one for each input state

        (See instantiate_mechanism_state_list for additional details)

        :param context:
        :return:
        """
        from Functions.MechanismStates.MechanismInputState import MechanismInputState

        self.inputStates = self.instantiate_mechanism_state_list(state_type=MechanismInputState,
                                                                  state_param_identifier=kwMechanismInputStates,
                                                                  constraint_values=self.variable,
                                                                  constraint_values_name="execute method variable",
                                                                  context=context)

        # Initialize self.inputValue to correspond to format of Mechanism's variable, and zero it
# FIX: INSURE THAT ELEMENTS CAN BE FLOATS HERE:  GET AND ASSIGN SHAPE RATHER THAN COPY? XXX
        self.inputValue = self.variable.copy() * 0.0

        # Assign self.inputState to first inputState in dict
        self.inputState = list(self.inputStates.values())[0]

    def instantiate_execute_method_parameter_states(self, context=NotImplemented):
        """Call instantiate_mechanism_state_list to instantiate MechanismParameterStates for subclass' execute method

        Instantiate parameter states for execute method params specified in kwExecuteMethodParams
        Use constraints (for compatibility checking) from paramsCurrent (inherited from paramClassDefaults)

        :param context:
        :return:
        """

        try:
            execute_method_param_specs = self.paramsCurrent[kwExecuteMethodParams]
        except KeyError:
            # No need to warn, as that already occurred in validate_params (above)
            return
        else:
            try:
                parameter_states = execute_method_param_specs[kwMechanismParameterStates]
            except KeyError:
                # kwMechanismParameterStates not specified, so continue
                pass
            else:
                # kwMechanismParameterStates was set to None, so do not instantiate any parameterStates
                if not parameter_states:
                    del self.paramsCurrent[kwExecuteMethodParams][kwMechanismParameterStates]
                    return
                # kwMechanismParameterStates was set to something;  pass for now
                pass
                # TBI / IMPLEMENT: use specs to implement paramterStates below
                # Notes:
                # * executeMethodParams are still available in paramsCurrent;
                # # just no parameterStates instantiated for them.

            self.executeMethodParameterStates = {}
            for param_name, param_value in execute_method_param_specs.items():

                param_state_spec = param_value

                from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
                self.executeMethodParameterStates[param_name] = self.instantiate_mechanism_state(
                                                                                state_type=MechanismParameterState,
                                                                                state_name=param_name,
                                                                                state_spec=param_state_spec,
                                                                                constraint_values=param_state_spec,
                                                                                constraint_values_name=param_name,
                                                                                context=context)

    def instantiate_output_states(self, context=NotImplemented):
        """Call instantiate_mechanism_state_list to instantiate orderedDict of outputState(s)

        Create OrderedDict of outputState(s) specified in paramsCurrent[kwMechanismInputStates]
        If kwMechanismInputStates is not specified, use self.variable to create a default output state
        When completed:
            - self.outputStates contains an OrderedDict of one or more outputStates
            - self.outputState contains first or only outputState in OrderedDict
            - paramsCurrent[kwMechanismOutputStates] contains the same OrderedDict (of one or more outputStates)
            - each outputState corresponds to an item in the output of the mechanism's execute method (EMO)
            - if there is only one outputState, it is assigned the full value

        (See instantiate_mechanism_state_list for additional details)

        IMPLEMENTATION NOTE:
            default(s) for self.paramsCurrent[kwMechanismOutputStates] (kwExecuteOutputDefault) is assigned here
            rather than in validate_params, as it requires execute method to have been instantiated first

        :param context:
        :return:
        """
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState

        self.outputStates = self.instantiate_mechanism_state_list(state_type=MechanismOutputState,
                                                                  state_param_identifier=kwMechanismOutputStates,
                                                                  constraint_values=self.value,
                                                                  constraint_values_name="execute method output",
                                                                  context=context)
        # Assign self.outputState to first outputState in dict
        self.outputState = list(self.outputStates.values())[0]

    def instantiate_mechanism_state_list(self,
                               state_type,              # MechanismStateType subclass
                               state_param_identifier,  # used to specify state_type state(s) in params[]
                               constraint_values,       # value(s) used as default for state and to check compatibility
                               constraint_values_name,  # name of constraint_values type (e.g. variable, output...)
                               context=NotImplemented):
        """Instantiate and return an OrderedDictionary of MechanismStates specified in paramsCurrent

        Arguments:
        - state_type (class): MechanismState class to be instantiated
        - state_param_identifier (str): kw used to identify set of states in paramsCurrent;  must be one of:
            - kwMechanismInputState
            - kwMechanismOutputState
# MODIFIED: 6/11/16 OLD:
        # - constraint_values (list): list of 1D np.ndarrays used as default values and for compatibility testing
        #     in instantiation of state(s):
        #     - kwMechanismInputState: self.variable
        #     - kwMechanismOutputState: self.value
        #     ?? ** Note:
        #     * this is ignored if param turns out to be a dict (entry value used instead)
        #                                           ?? TEST THIS
# MODIFIED: 6/11/16 NEW:

        - constraint_values (2D np.array): set of 1D np.ndarrays used as default values and
            for compatibility testing in instantiation of state(s):
            - kwMechanismInputState: self.variable
            - kwMechanismOutputState: self.value
            ?? ** Note:
            * this is ignored if param turns out to be a dict (entry value used instead)
# MODIFIED: 6/11/16 END:
        - constraint_values_name (str):  passed to instantiate_mechanism_state, used in error messages
        - context (str)

        If state_param_identifier is absent from paramsCurrent:
            - instantiate a default MechanismState using constraint_values,
            - place as the single entry in the OrderedDict
        Otherwise, if the param(s) in paramsCurrent[state_param_identifier] is/are:
            - a single value:
                instantiate it (if necessary) and place as the single entry in an OrderedDict
            - a list:
                instantiate each item (if necessary) and place in an OrderedDict
            - an OrderedDict:
                instantiate each entry value (if necessary)
        In each case, generate an OrderedDict with one or more entries, assigning:
            the key for each entry the name of the outputState if provided,
                otherwise, use kwMechanism<state_type>States-n (incrementing n for each additional entry)
            the state value for each entry to the corresponding item of the mechanism's state_type state's value
            the dict to both self.<state_type>States and paramsCurrent[kwMechanism<state_type>States]
            self.<state_type>State to self.<state_type>States[0] (the first entry of the dict)
        Notes:
            * if there is only one state, but the value of the mechanism's state_type has more than one item:
                assign it to the sole state, which is assumed to have a multi-item value
            * if there is more than one state:
                the number of states must match length of mechanisms state_type value or an exception is raised

        :param state_type:
        :param state_param_identifier:
        :param constraint_values:
        :param constraint_values_name:
        :param context:
        :return:
        """

        state_entries = self.paramsCurrent[state_param_identifier]

        # If kwMechanism<*>States was found to be absent or invalid in validate_params, it was set to None there
        #     for instantiation (here) of a default state_type MechanismState using constraint_values for the defaults
        if not state_entries:
            # assign constraint_values as single item in a list, to be used as state_spec below
# MODIFIED 6/13/16 OLD:
#             state_entries = [constraint_values]
# MODIFIED 6/13/16 NEW:
            state_entries = constraint_values
# MODIFIED END
            # issue warning if in VERBOSE mode:
            if self.prefs.verbosePref:
                print("No {0} specified for {1}; default will be created using {2} of execute method ({3})"
                      " as its value".format(state_param_identifier,
                                             self.__class__.__name__,
                                             constraint_values_name,
                                             constraint_values))

# MODIFIED 6/13/16 OLD:
#         # kwMechanism<*>States should now be either a list (possibly constructed in validate_params) or an OrderedDict:
#         if isinstance(state_entries, (list, OrderedDict)):
# MODIFIED 6/13/16 NEW:
        # kwMechanism<*>States should now be either a list (possibly constructed in validate_params) or an OrderedDict:
        if isinstance(state_entries, (list, OrderedDict, np.ndarray)):
# MODIFIED 6/13/16 END

            num_states = len(state_entries)

            # Check that constraint_values is an indexible object, the items of which are the constraints for each state
            # Notes
            # * generally, this will be a list or an np.ndarray (either >= 2D np.array or with a dtype=object)
            # * for MechanismOutputStates, this should correspond to its value
            try:
                # Insure that constraint_values is an indexible item (list, >=2D np.darray, or otherwise)
                num_constraint_items = len(constraint_values)
            except:
                raise MechanismError("PROGRAM ERROR: constraint_values ({0}) for {1} of {2}"
                                     " must be an indexible object (e.g., list or np.ndarray)".
                                     format(constraint_values, constraint_values_name, state_type.__name__))

            # # If there is only one state, assign full constraint_values to sole state
            # #    but only do this if number of constraints is > 1, as need to leave solo exposed value intact
            # if num_states == 1 and num_constraint_items > 1:
            #     state_constraint_value = [constraint_values]

            # If number of states exceeds number of items in constraint_values, raise exception
            if num_states > num_constraint_items:
                raise MechanismError("There are too many {0} specified ({1}) in {2} "
                                     "for the number of values ({3}) in the {4} of its execute method".
                                     format(state_param_identifier,
                                            num_states,
                                            self.__class__.__name__,
                                            num_constraint_items,
                                            constraint_values_name))

            # If number of states is less than number of items in constraint_values, raise exception
            elif num_states < num_constraint_items:
                raise MechanismError("There are fewer {0} specified ({1}) than the number of values ({2}) "
                                     "in the {3} of the execute method for {4}".
                                     format(state_param_identifier,
                                            num_states,
                                            num_constraint_items,
                                            constraint_values_name,
                                            self.name))

            # Iterate through list or state_dict:
            # - instantiate each item or entry as state_type MechanismState
            # - get name, and use as key to assign as entry in self.<*>states
            states = OrderedDict()

            # Instantiate state for entry in list or dict
            # Note: if state_entries is a list, state_spec is the item, and key is its index in the list
            for key, state_spec in state_entries if isinstance(state_entries, dict) else enumerate(state_entries):
                state_name = ""

                # If state_entries is already an OrderedDict, then use:
                # - entry key as state's name
                # - entry value as state_spec
                if isinstance(key, str):
                    state_name = key
                    state_constraint_value = constraint_values
                    # Note: state_spec has already been assigned to entry value by enumeration above

                # If state_entries is a list, and item is a string, then use:
                # - string as the name for a default state
                # - key (index in list) to get corresponding value from constraint_values as state_spec
                # - assign same item of contraint_values as the constraint
                elif isinstance(state_spec, str):
                    # Use state_spec as state_name if it has not yet been used
                    if not state_name is state_spec and not state_name in states:
                        state_name = state_spec
                    # Add index suffix to name if it is already been used
                    # Note: avoid any chance of duplicate names (will cause current state to overwrite previous one)
                    else:
                        state_name = state_spec + '-' + str(key)
                    state_spec = constraint_values[key]
                    state_constraint_value = constraint_values[key]

                # If state entries is a list, but item is NOT a string, then:
                # - use default name (which is incremented for each instance in register_categories)
                # - use item as state_spec (i.e., assume it is a specification for a MechanismState)
                #   Note:  still need to get indexed element of constraint_values,
                #          since it was passed in as a 2D array (one for each state)
                else:
                    # If only one state, don't add index suffix
                    if num_states == 1:
                        state_name = 'Default' + state_param_identifier[:-1]
                    # Add incremented index suffix for each state name
                    else:
                        state_name = 'Default' + state_param_identifier[:-1] + "-" + str(key+1)
                    # Note: state_spec has already been assigned to item in state_entries list by enumeration above
                    state_constraint_value = constraint_values[key]

                state = self.instantiate_mechanism_state(state_type=state_type,
                                                         state_name=state_name,
                                                         state_spec=state_spec,
                                                         constraint_values=state_constraint_value,
                                                         constraint_values_name=constraint_values_name,
                                                         context=context)
                # Get name of state, and use as key to assign to states OrderedDict
                states[state.name] = state
            return states

        else:
            # This shouldn't happen, as kwMechanism<*>States was validated to be one of the above in validate_params
            raise MechanismError("Program error: {0} for is not a recognized {1} specification for {2}; "
                                 "it should have been converted to a list in Mechanism.validate_params)".
                                 format(state_entries, state_param_identifier, self.__class__.__name__))

    def instantiate_mechanism_state(self,
                                    state_type,            # MechanismState subclass
                                    state_name,            # Name used to refer to subclass in prompts
                                    state_spec,            # MechanismStateType subclass, object, spec dict or value
                                    constraint_values,     # Value used to check compatibility
                                    constraint_values_name, # Name of constraint_values's type (e.g. variable, output...)
                                    # constraint_index=NotImplemented, # Used in cases of multiple states
                                    context=NotImplemented):
        """Instantiate a MechanismState of specified type, with a value that is compatible with constraint_values

        Constraint value must be a number or a list or tuple of numbers
        (since it is used as the variable for instantiating the requested state)

        If state_spec is a:
        + MechanismState class:
            implements default using constraint_values
        + MechanismState object:
            checks ownerMechanism is owner (if not, user is given options in check_mechanism_state_ownership)
            checks compatibility of value with constraint_values
        + Projection object:
            assigns constraint_values to value
            assigns projection to kwMechanismStateParams{kwMechanismStateProjections:<projection>}
        + Projection class (or keyword string constant for one):
            assigns constraint_values to value
            assigns projection class spec to kwMechanismStateParams{kwMechanismStateProjections:<projection>}
        + specification dict for MechanismState (see XXX for context):
            check compatibility of kwMechanismStateValue with constraint_values
        + ParamValueProjection tuple: (only allowed for MechanismParameterState spec)
            assigns ParamValueProjection.value to state_spec
            assigns ParamValueProjection.projection to kwMechanismStateParams{kwMechanismStateProjections:<projection>}
        + 2-item tuple: (only allowed for MechanismParameterState spec)
            assigns first item to state_spec
            assigns second item to kwMechanismStateParams{kwMechanismStateProjections:<projection>}
        + value:
            checks compatibility with constraint_values
        If any of the conditions above fail:
            a default MechanismState of specified type is instantiated using constraint_values as value

        :param context: (str)
        :return mechanismState: (MechanismState)
        """

# IMPLEMENTATION NOTE: CONSIDER MOVING MUCH IF NOT ALL OF THIS TO MechanismState.__init__()

        #region VALIDATE ARGS
        if not inspect.isclass(state_type) or not issubclass(state_type, MechanismState):
            raise MechanismError("state_type arg ({0}) to instantiate_mechanism_state "
                                 "must be a MechanismState subclass".format(state_type))
        if not isinstance(state_name, str):
            raise MechanismError("state_name arg ({0}) to instantiate_mechanism_state must be a string".
                                 format(state_name))
        if not isinstance(constraint_values_name, str):
            raise MechanismError("constraint_values_name arg ({0}) to instantiate_mechanism_state must be a string".
                                 format(constraint_values_name))
        #endregion

        # Assume state is specified as a value, so set state_value to it; if otherwise, will be overridden below
        state_value = state_spec
        state_params = {}
        # Used locally to report type of specification for MechanismState
        #  if value is not compatible with constraint_values
        spec_type = None

        #region CHECK FORMAT OF constraint_values AND CONVERT TO SIMPLE VALUE
        # If constraint_values is a class:
        if inspect.isclass(constraint_values):
            # If constraint_values is a MechanismState class, set to variableClassDefault:
            if issubclass(constraint_values, MechanismState):
                constraint_values = state_spec.variableClassDefault
            # If constraint_values is a Projection, set to output of execute method:
            if issubclass(constraint_values, Projection):
                constraint_values = constraint_values.value
        # If constraint_values is a MechanismState object, set to value:
        elif isinstance(constraint_values, state_type):
            constraint_values = constraint_values.value
        # If constraint_values is a specification dict, presumably it is for a MechanismState:
        elif isinstance(constraint_values, dict):
            constraint_values = constraint_values[kwMechanismStateValue]
        # If constraint_values is a ParamValueProjection tuple, set to ParamValueProjection.value:
        elif isinstance(constraint_values, ParamValueProjection):
            constraint_values = constraint_values.value
        # Otherwise, assumed to be a value

        # # MODIFIED 6/14/16: QQQ - WHY DOESN'T THIS WORK HERE?? (DONE BELOW, JUST BEFORE CALLING state = state_type(<>)
        # # CONVERT CONSTRAINT_VALUES TO NP ARRAY AS ACTUAL STATE VALUES WILL BE SO CONVERTED (WHERE ??)
        # #  Convert constraint_values to np.array as actual state value is converted
        # constraint_values = convert_to_np_array(constraint_values,1)
        # # MODIFIED END

        #endregion

        #region CHECK COMPATIBILITY OF state_spec WITH constraint_values

        # MechanismState subclass
        # If state_spec is a subclass:
        # - instantiate default using constraint_values as value
        if inspect.isclass(state_spec) and issubclass(state_spec, state_type):
            state_value = constraint_values

        # MechanismState object
        # If state_spec is a MechanismState object:
        # - check that its value attribute matches the constraint_values
        # - check that its ownerMechanism = owner
        # - if either fails, assign default
        # from Functions.MechanismStates.MechanismState import MechanismOutputState
        if isinstance(state_spec, state_type):
            # Check that MechanismState's value is compatible with Mechanism's variable
            if iscompatible(state_spec.value, constraint_values):
                # Check that Mechanism is MechanismState's owner;  if it is not, user is given options
                state =  self.check_mechanism_state_ownership(state_name, state_spec)
                if state:
                    return
                else:
                    # MechanismState was rejected, and assignment of default selected
                    state = constraint_values
            else:
                # MechanismState's value doesn't match constraint_values, so assign default
                state = constraint_values
                spec_type = state_name

        # Specification dict
        # If state_spec is a specification dict:
        # - check that kwMechanismStateValue matches constraint_values and assign to state_value
        # - assign kwMechanismState params to state_params
        if isinstance(state_spec, dict):
            try:
                state_value =  state_spec[kwMechanismStateValue]
            except KeyError:
                state_value = constraint_values
                if self.prefs.verbosePref:
                    print("{0} missing from inputState specification dict for {1};  default ({2}) will be used".
                                         format(kwMechanismStateValue, self.name, constraint_values))
            if not iscompatible(state_value, constraint_values):
                state_value = constraint_values
                spec_type = kwMechanismStateValue
            try:
                state_params = state_spec[kwMechanismStateParams]
            except KeyError:
                state_params = {}

        # ParamValueProjection
        # If state_type is MechanismParameterState and state_spec is a ParamValueProjection tuple:
        # - check that ParamValueProjection.value matches constraint_values and assign to state_value
        # - assign ParamValueProjection.projection to kwMechanismStateParams:{kwMechanismStateProjections:<projection>}
        # Note: validity of projection specification or compatiblity of projection's variable or execute method output
        #       with state value is handled in MechanismState.instantiate_projections
        if isinstance(state_spec, ParamValueProjection):
            from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
            if not issubclass(state_type, MechanismParameterState):
                raise MechanismError("ParamValueProjection ({0}) not permitted as specification for {1} (in {2})".
                                     format(state_spec, state_type.__name__, self.name))
            state_value =  state_spec.value
            if not iscompatible(state_value, constraint_values):
                state_value = constraint_values
                spec_type = 'ParamValueProjection'
            state_params = {kwMechanismStateProjections:[state_spec.projection]}

        # 2-item tuple (param_value, projection_spec) [convenience notation for projection to parameterState]:
        # If state_type is MechanismParameterState, and state_spec is a tuple with two items, the second of which is a
        #    projection specification (kwControlSignal or kwMapping)), allow it (though should use ParamValueProjection)
        # - check that first item matches constraint_values and assign to state_value
        # - assign second item as projection to kwMechanismStateParams:{kwMechanismStateProjections:<projection>}
        # Note: validity of projection specification or compatibility of projection's variable or execute method output
        #       with state value is handled in MechanismState.instantiate_projections
        # IMPLEMENTATION NOTE:
        #    - need to do some checking on state_spec[1] to see if it is a projection
        #      since it could just be a numeric tuple used for the variable of a mechanismState;
        #      could check string against ProjectionRegistry (as done in parse_projection_ref in MechanismState)
        if (isinstance(state_spec, tuple) and len(state_spec) is 2 and
                (state_spec[1] is kwControlSignal or
                         state_spec[1] is kwMapping or
                     isinstance(state_spec[1], Projection) or
                     inspect.isclass(state_spec[1] and issubclass(state_spec[1], Projection))
                 )):
            from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
            if not issubclass(state_type, MechanismParameterState):
                raise MechanismError("Tuple with projection spec ({0}) not permitted as specification for {1} (in {2})".
                                     format(state_spec, state_type.__name__, self.name))
            state_value =  state_spec[0]
            constraint_values = state_value
            # if not iscompatible(state_value, constraint_values):
            #     state_value = constraint_values
            #     spec_type = 'ParamValueProjection'
            state_params = {kwMechanismStateProjections:[state_spec[1]]}

        # Projection
        # If state_spec is a Projection object or Projection class
        # - assign constraint_values to state_value
        # - assign ParamValueProjection.projection to kwMechanismStateParams:{kwMechanismStateProjections:<projection>}
        # Note: validity of projection specification or compatibility of projection's variable or execute method output
        #       with state value is handled in MechanismState.instantiate_projections
        try:
            issubclass(state_spec, Projection)
        except TypeError:
            if isinstance(state_spec, (Projection, str)):
                state_value =  constraint_values
                state_params = {kwMechanismStateProjections:{kwProjectionType:state_spec}}
        else:
            state_value =  constraint_values
            state_params = {kwMechanismStateProjections:state_spec}

        # FIX:  WHEN THERE ARE MULTIPLE STATES, LENGTH OF constraint_values GROWS AND MISMATCHES state_value
        # IMPLEMENT:  NEED TO CHECK FOR ITEM OF constraint_values AND CHECK COMPATIBLITY AGAINST THAT
        #         # Do one last check for compatibility of value with constraint_values (in case state_spec was a value)
        if not iscompatible(state_value, constraint_values):
            # FIX:  IMPLEMENT TEST OF constraint_index HERE 5/26/16
            # pass
            state_value = constraint_values
            spec_type = state_name

        # WARN IF DEFAULT (constraint_values) HAS BEEN ASSIGNED
        # spec_type has been assigned, so iscompatible() failed above and constraint value was assigned
        if spec_type:
            if self.prefs.verbosePref:
                print("Value ({0}) of {1} (type: {2}) is not compatible with {3} ({4}) of {6};"
                      " default {4} will be created using {5}".
                      format(state_value,
                             state_name,
                             spec_type,
                             constraint_values_name,
                             constraint_values.__class__.__name__,
                             constraint_values,
                             self.__class__.__name__))
        #endregion

        #region INSTANTIATE STATE:
        # Instantiate new MechanismState
        # Note: this will be either a default MechanismState instantiated using constraint_values as its value
        #       or one determined by a specification dict, depending on which of the following obtained above:
        # - state_spec was a ParamValueProjection tuple
        # - state_spec was a specification dict
        # - state_spec was a value
        # - value of specified MechanismState was incompatible with constraint_values
        # - owner of MechanismState was not self and user chose to implement default
        # IMPLEMENTATION NOTE:
        # - setting prefs=NotImplemented causes TypeDefaultPreferences to be assigned (from FunctionPreferenceSet)
        # - alternative would be prefs=self.prefs, causing state to inherit the prefs of its ownerMechanism;

        #  Convert constraint_values to np.array to match state_value (which, as output of execute method, will be one)
        constraint_values = convert_to_np_array(constraint_values,1)

        # Implement default MechanismState
        state = state_type(owner_mechanism=self,
                           reference_value=constraint_values,
                           value=state_value,
                           name=state_name,
                           params=state_params,
                           prefs=NotImplemented,
                           context=context)

# FIX LOG: ADD NAME TO LIST OF MECHANISM'S VALUE ATTRIBUTES FOR USE BY LOGGING ENTRIES
        # This is done here to register name with Mechanism's stateValues[] list
        # It must be consistent with value setter method in MechanismState
# FIX LOG: MOVE THIS TO MECHANISM STATE __init__ (WHERE IT CAN BE KEPT CONSISTENT WITH setter METHOD??
#      OR MAYBE JUST REGISTER THE NAME, WITHOUT SETTING THE
        setattr(self, state.name+'.value', state.value)

        #endregion

        return state

    def check_mechanism_parameter_state_value(self, param_name, value):
        """Check that parameter value (<MechanismParameterState>.value) is compatible with value in paramClassDefault

        :param param_name: (str)
        :param value: (value)
        :return: (value)
        """
        default_value = self.paramClassDefaults[param_name]
        if iscompatible(value, default_value):
            return value
        else:
            if self.prefs.verbosePref:
                print("Format is incorrect for value ({0}) of {1} in {2};  default ({3}) will be used.".
                      format(value, param_name, self.name, default_value))
            return default_value

    def check_mechanism_state_ownership(self, param_name, mechanism_state):
        """Check whether MechanismState's owner is self and if not offer options how to handle it

        If MechanismState's owner is not self, options offered to:
        - reassign it to self
        - make a copy and assign to self
        - return None => caller should assign default

        :param param_name: (str)
        :param mechanism_state: (MechanismState)
        :param context: (str)
        :return: (MechanismState or None)
        """

        if mechanism_state.ownerMechanism != self:
            reassign = input("\nMechanismState {0}, assigned to {1} in {2}, already belongs to {3}"
                             " You can choose to reassign it (r), copy it (c), or assign default (d):".
                             format(mechanism_state.name, param_name, self.name,
                                    mechanism_state.ownerMechanism.name))
            while reassign != 'r' and reassign != 'c' and reassign != 'd':
                reassign = input("\nReassign (r), copy (c), or default (d):".
                                 format(mechanism_state.name, param_name, self.name,
                                        mechanism_state.ownerMechanism.name))

                if reassign == 'r':
                    while reassign != 'y' and reassign != 'n':
                        reassign = input("\nYou are certain you want to reassign it {0}? (y/n):".
                                         format(param_name))
                    if reassign == 'y':
                        # Note: assumed that parameters have already been checked for compatibility with assignment
                        return mechanism_state

            # Make copy of mechanismState
            if reassign == 'c':
                import copy
                mechanism_state = copy.deepcopy(mechanism_state)

            # Assign self to chosen mechanismState
            mechanism_state.ownerMechanism = self
        return mechanism_state

    def add_projection_to_mechanism(self, projection, state, context=NotImplemented):
        """Add projection to specified state

        projection can be any valid specification of a projection (see MechanismState.instantiate_projections)
        state must be a specification of a MechanismInputState or MechanismParameterState
        Specification of MechanismInputState can be any of the following:
                - kwMechanismInputState - assigns projection to (primary) inputState
                - MechanismInputState object
                - index for Mechanism.inputStates OrderedDict
                - name of inputState (i.e., key for Mechanism.inputStates OrderedDict))
                - the keyword kwAddMechanismInputState or the name for an inputState to be added
        Specification of MechanismParameterState must be a MechanismParameterState object
        IMPLEMENTATION NOTE:  ADD FULL SET OF MechanismParameterState SPECIFICATIONS

        Args:
            projection:
            input_state:

        """
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
        if not isinstance(state, (int, str, MechanismInputState, MechanismParameterState)):
            raise MechanismError("State specification(s) for {0} (as receivers of {1}) contain(s) one or more items"
                                 " that is not a name, reference to an inputState or parameterState object, "
                                 " or an index (for inputStates)".
                                 format(self.name, projection))

        # state is MechanismState object, so use that
        if isinstance(state, MechanismState):
            state.instantiate_projections(projections=projection, context=context)
            return

        # Generic kwMechanismInputState is specified, so use (primary) inputState
        elif state is kwMechanismInputState:
            self.inputState.instantiate_projections(projections=projection, context=context)
            return

        # input_state is index into inputStates OrderedDict, so get corresponding key and assign to input_state
        elif isinstance(state, int):
            try:
                key = list(self.inputStates.keys)[state]
            except IndexError:
                raise MechanismError("Attempt to assign projection ({0}) to inputState {1} of {2} "
                                     "but it has only {3} inputStates".
                                     format(projection.name, state, self.name, len(self.inputStates)))
            else:
                input_state = key

        # input_state is string (possibly key retrieved above)
        #    so try as key in inputStates OrderedDict (i.e., as name of an inputState)
        if isinstance(state, str):
            try:
                self.inputState[state].instantiate_projections(projections=projection, context=context)
            except KeyError:
                pass
            else:
                if self.prefs.verbosePref:
                    print("Projection {0} added to {1} of {2}".format(projection.name, state, self.name))
                # return

        # input_state is either the name for a new inputState or kwAddNewMechanismInputState
        if not state is kwAddMechanismInputState:
            if self.prefs.verbosePref:
                reassign = input("\nAdd new inputState named {0} to {1} (as receiver for {2})? (y/n):".
                                 format(input_state, self.name, projection.name))
                while reassign != 'y' and reassign != 'n':
                    reassign = input("\nAdd {0} to {1}? (y/n):".format(input_state, self.name))
                if reassign == 'n':
                    raise MechanismError("Unable to assign projection {0} to {1}".format(projection.name, self.name))

        input_state = self.instantiate_mechanism_state(
                                        state_type=MechanismInputState,
                                        state_name=input_state,
                                        state_spec=projection.value,
                                        constraint_values=projection.value,
                                        constraint_values_name='Projection value for new inputState',
                                        context=context)
            #  Update inputState and inputStates
        try:
            self.inputStates[input_state.name] = input_state
        # No inputState(s) yet, so create them
        except AttributeError:
            self.inputStates = OrderedDict({input_state.name:input_state})
            self.inputState = list(self.inputStates)[0]
        input_state.instantiate_projections(projections=projection, context=context)

    def add_projection_from_mechanism(self, projection, state, context=NotImplemented):
    # IMPLEMENTATION NOTE: TBI
        """Add projection to specified state
        """
        pass

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Update inputState(s) and param(s), call subclass executeMethod, update outputState(s), and assign self.value

        Arguments:
        - time_scale (TimeScale): time scale at which to run subclass execute method
        - params (dict):  params for subclass execute method (overrides its paramInstanceDefaults and paramClassDefaults
        - context (str): should be set to subclass name by call to super from subclass

        Execution sequence:
        • Call self.inputState.execute() for each entry in self.inputStates:
            + execute every self.inputState.receivesFromProjections.[<Projection>.execute()...]
            + aggregate results using self.inputState.params[kwExecuteMethod]()
            + store the result in self.inputState.value
        • Call every self.params[<MechanismParameterState>].execute(); for each:
            + execute self.params[<MechanismParameterState>].receivesFromProjections.[<Projection>.execute()...]
                (usually this is just a single ControlSignal)
            + aggregate results (if > one) using self.params[<MechanismParameterState>].params[kwExecuteMethod]()
            + apply the result to self.params[<MechanismParameterState>].value
        • Call subclass' self.execute(params):
            - use self.inputState.value as its variable,
            - use params[kw<*>] or self.params[<MechanismParameterState>].value for each param of subclass self.execute,
            - apply the output to self.outputState.value
            Note:
            * if execution is occuring as part of initialization, outputState(s) are reset to 0
            * otherwise, they are left in the current state until the next update

        • [TBI: Call self.outputState.execute() (output gating) to update self.outputState.value]

        Returns self.outputState.value and self.outputStates[].value after either one time_step or the full trial
             (set by params[kwMechanismTimeScale)

        :param self:
        :param params: (dict)
        :param context: (optional)
        :rtype outputState.value (list)
        """

        # IMPLEMENTATION NOTE: Re-write by calling execute methods according to order executeMethodDict:
        #         for func in self.executeMethodDict:
        #             self.executeMethodsDict[func]()

        #region VALIDATE RUNTIME PARAMETER SETS
        # Insure that param set is for a MechanismStates:
        if self.prefs.paramValidationPref:
            # if runtime_params != NotImplemented:
            if runtime_params:
                for param_set in runtime_params:
                    if not (kwMechanismInputStateParams in param_set or
                            kwMechanismParameterStateParams in param_set or
                            kwMechanismOutputStateParams in param_set):
                        raise MechanismError("{0} is not a valid parameter set for run specification".format(param_set))
        #endregion

        #region VALIDATE INPUT STATE AND RUNTIME PARAMS
        self.check_args(variable=self.inputState.value,
                        params=runtime_params,
                        target_set=runtime_params)
        #endregion

        #region UPDATE INPUT STATE(S)
        self.update_input_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region UPDATE PARAMETER STATE(S)
        self.update_parameter_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region CALL executeMethod AND ASSIGN RESULT TO self.value
# CONFIRM: VALIDATION METHODS CHECK THE FOLLOWING CONSTRAINT: (AND ADD TO CONSTRAINT DOCUMENTATION):
# DOCUMENT: #OF OUTPUTSTATES MUST MATCH #ITEMS IN OUTPUT OF EXECUTE METHOD **
#         # MODIFIED 7/9/16 OLD:
#         self.value = self.execute(time_scale=time_scale, context=context)
        # MODIFIED 7/9/16 NEW:
        self.value = self.execute(variable=self.inputValue, time_scale=time_scale, context=context)
        #endregion

        #region UPDATE OUTPUT STATE(S)
        self.update_output_states(time_scale=time_scale, context=context)
        #endregion

        #region TBI
        # # Call outputState.execute
        # #    updates outState.value, based on any projections (e.g., gating) it may get
        # self.inputState.execute()
        #endregion

        #region REPORT EXECUTION
        import re
        # if (self.prefs.reportOutputPref and not (context is NotImplemented or kwFunctionInit in context)):
        if self.prefs.reportOutputPref and not context is NotImplemented and kwExecuting in context:
            print("\n{0} Mechanism executed:\n- output: {1}".
                  format(self.name, re.sub('[\[,\],\n]','',str(self.outputState.value))))
        # if self.prefs.reportOutputPref and not context is NotImplemented and kwEVCSimulation in context:
        #     print("\n{0} Mechanism simulated:\n- output: {1}".
        #           format(self.name, re.sub('[\[,\],\n]','',str(self.outputState.value))))

        #endregion

        #region RE-SET STATE_VALUES AFTER INITIALIZATION
        # If this is (the end of) an initialization run, restore state values to initial condition
        if '_init_' in context:
            for state in self.inputStates:
                self.inputStates[state].value = self.inputStates[state].variable
            for state in self.executeMethodParameterStates:
                self.executeMethodParameterStates[state].value =  self.executeMethodParameterStates[state].baseValue
            for state in self.outputStates:
                # Zero outputStates in case of recurrence:
                #    don't want any non-zero values as a residuum of initialization runs to be
                #    transmittted back via recurrent projections as initial inputs
# FIX: IMPLEMENT zero_all_values METHOD
                self.outputStates[state].value = self.outputStates[state].value * 0.0
        #endregion

        #endregion

        # MODIFIED 7/9/16 OLD:
        # return self.outputState.value
        # MODIFIED 7/9/16 NEW:
        return self.value


    def update_input_states(self, runtime_params=NotImplemented, time_scale=NotImplemented, context=NotImplemented):
        """ Update value for each inputState in self.inputStates:

        Call execute method for all (Mapping) projections in inputState.receivesFromProjections
        Aggregate results (using inputState execute method)
        Update inputState.value

        Args:
            params:
            time_scale:
            context:

        Returns:
        """
        for i in range(len(self.inputStates)):
            state_name, state = list(self.inputStates.items())[i]
            state.update(params=runtime_params, time_scale=time_scale, context=context)
            self.inputValue[i] = state.value
        self.variable = np.array(self.inputValue)

    def update_parameter_states(self, runtime_params=NotImplemented, time_scale=NotImplemented, context=NotImplemented):
        for state_name, state in self.executeMethodParameterStates.items():
            state.update(params=runtime_params, time_scale=time_scale, context=context)

    def update_output_states(self, time_scale=NotImplemented, context=NotImplemented):
        """Assign value of each outputState in outputSates

        Assign each item of self.execute's return value to the value of the corresponding outputState in outputSates

        IMPLEMENTATION NOTE:

        """
        # MODIFIED 7/9/16 OLD:
        # # FIX: ??CONVERT OUTPUT TO 2D ARRAY HERE??
        # output = self.execute(time_scale=time_scale, context=context)
        # # output = np.atleast_2d(self.execute(time_scale=time_scale, context=context))
        # for state in self.outputStates:
        #     i = list(self.outputStates.keys()).index(state)
        #     self.outputStates[state].value = output[i]

        # MODIFIED 7/9/16 NEW:  [MOVED CALL TO self.execute TO Mechanism.update() AND REPLACED output WITH self.value]
        # FIX: ??CONVERT OUTPUT TO 2D ARRAY HERE??
        for state in self.outputStates:
            i = list(self.outputStates.keys()).index(state)
            self.outputStates[state].value = self.value[i]

    def execute(self, variable, params, time_scale, context):
        raise MechanismError("{0} must implement execute method".format(self.__class__.__name__))

    def adjust_function(self, params, context=NotImplemented):
        """Modify control_signal_allocations while process is executing

        called by process.adjust()
        returns output after either one time_step or full trial (determined by current setting of time_scale)

        :param self:
        :param params: (dict)
        :param context: (optional)
        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """

        self.assign_defaults(self.inputState, params)
# IMPLEMENTATION NOTE: *** SHOULD THIS UPDATE AFFECTED PARAM(S) BY CALLING self.update_parameter_states??
        return self.outputState.value

    def terminate_execute(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        if context==NotImplemented:
            raise MechanismError("terminate execute method not implemented by mechanism sublcass")

    def get_mechanism_param_values(self):
        """Return dict with current value of each MechanismParameterState in paramsCurrent
        :return: (dict)
        """
        from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
        return dict((param, value.value) for param, value in self.paramsCurrent.items()
                    if isinstance(value, MechanismParameterState) )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):

        self._value = assignment
# TEST:
        test = self.value
        temp = test
    @property
    def inputState(self):
        return self._inputState

    @inputState.setter
    def inputState(self, assignment):
        self._inputState = assignment

    @property
    def outputState(self):
        return self._outputState

    @outputState.setter
    def outputState(self, assignment):
        self._outputState = assignment


# **************************************** SystemDefaultMechanism ******************************************************


class SystemDefaultMechanism_Base(Mechanism_Base):
    """Use to implement SystemDefaultInputMechanism, SystemDefaultControlMechanism, and SystemDefaultOutputMechanism

    Description:
        Implements "dummy" mechanism used to implement default input, control signals, and outputs to other mechanisms

    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwMechanismInputStateValue: [0]
            # + kwMechanismOutputStateValue: [1]
            + kwExecuteMethod: Linear
            + kwExecuteMethodParams:{kwSlope:1, kwIntercept:0}
    """

    functionType = "SystemDefaultMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemDefaultMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    variableClassDefault = SystemDefaultInputValue

    from Functions.Utility import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwExecuteMethod:Linear,
        kwExecuteMethodParams:{Linear.kwSlope:1, Linear.kwIntercept:0},
    })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Add Linear as default executeMethod, assign default name, and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType

        self.functionName = self.functionType

        super(SystemDefaultMechanism_Base, self).__init__(variable=default_input_value,
                                                       params=params,
                                                       name=name,
                                                       prefs=prefs,
                                                       context=self)

