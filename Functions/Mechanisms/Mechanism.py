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
    ONLY_SPECIFIED_OUTPUT_STATES = ()
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
            - DefaultProcessingMechanism_Base (used for SystemDefaultInputMechanism and SystemDefaultOutputMechanism)
            - DDM (default Mechanism)
            - DefaultControlMechanism (used for DefaultController)

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
        - params (dict): (see validate_params below and MechanismState.instantiate_mechanism_state() for details)
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
                * note: inputStates can also be added using MechanismState.instantiate_mechanism_state()
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
                * note: inputStates can also be added using MechanismState.instantiate_mechanism_state()
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
        + variable (value): used as input to mechanism's execute method
        + paramsCurrent (dict): current value of all params for instance (created and validated in Functions init)
        + paramInstanceDefaults (dict): defaults for instance (created and validated in Functions init)
        + paramNames (list): list of keys for the params in paramInstanceDefaults
        + inputState (MechanismInputState): default MechanismInput object for mechanism
        + inputStates (dict): created if params[kwMechanismInputState] specifies  more than one MechanismInputState
        + inputValue (value, list or ndarray): value, list or array of values, one for the value of each inputState
        + receivesProcessInput (bool): flags if Mechanism (as first in Configuration) receives Process input projection
        + executeMethodParameterStates (dict): created if params[kwExecuteMethodParams] specifies any parameters
        + outputState (MechanismOutputState): default MechanismOutputState for mechanism
        + outputStates (dict): created if params[kwMechanismOutputStates] specifies more than one MechanismOutputState
        + value (value, list, or ndarray): output of the Mechanism's execute method;
            Note: currently each item of self.value corresponds to value of corresponding outputState in outputStates
        + outputStateValueMapping (dict): specifies index of each state in outputStates,
            used in update_output_states to assign the correct item of value to each outputState in outputStates
            Notes:
            * any Function with an executeMethod that returns a value with len > 1 MUST implement self.execute
            *    rather than just use the params[kwExecuteMethod] so that outputStateValueMapping can be implemented
            * TBI: if the executeMethod of a Function is specified only by params[kwExecuteMethod]
                       (i.e., it does not implement self.execute) and it returns a value with len > 1
                       it MUST also specify kwExecuteMethodOutputStateValueMapping
        + phaseSpec (int or float): time_step(s) on which Mechanism.update() is called (see Process for specification)
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, default is created by copying Mechanism_BasePreferenceSet

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
        # MODIFIED 7/16/16 NEW:
        kwMonitoredOutputStates:NotImplemented
        # MODIFIED END
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
            # MODIFIED 7/16/16 NEW:
            if not target_set[kwMonitoredOutputStates] or target_set[kwMonitoredOutputStates] is NotImplemented:
                pass
            # MODIFIED END
            # It is a MonitoredOutputStatesOption specification
            elif isinstance(target_set[kwMonitoredOutputStates], MonitoredOutputStatesOption):
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
        """Call MechanismState.instantiate_mechanism_state_list() to instantiate orderedDict of inputState(s)

        Create OrderedDict of inputState(s) specified in paramsCurrent[kwMechanismInputStates]
        If kwMechanismInputStates is not specified, use self.variable to create a default input state
        When completed:
            - self.inputStates contains an OrderedDict of one or more inputStates
            - self.inputState contains first or only inputState in OrderedDict
            - paramsCurrent[kwMechanismOutputStates] contains the same OrderedDict (of one or more inputStates)
            - each inputState corresponds to an item in the variable of the mechanism's execute method (EMV)
            - if there is only one inputState, it is assigned the full value

        Note: MechanismState.instantiate_mechanism_state_list()
                  parses self.variable (2D np.array, passed in constraint_values)
                  into individual 1D arrays, one for each input state

        (See MechanismState.instantiate_mechanism_state_list() for additional details)

        :param context:
        :return:
        """
        from Functions.MechanismStates.MechanismState import instantiate_mechanism_state_list
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        self.inputStates = instantiate_mechanism_state_list(owner=self,
                                                            state_list=self.paramsCurrent[kwMechanismInputStates],
                                                            state_type=MechanismInputState,
                                                            state_param_identifier=kwMechanismInputStates,
                                                            constraint_values=self.variable,
                                                            constraint_values_name="execute method variable",
                                                            context=context)

        # Initialize self.inputValue to correspond to format of Mechanism's variable, and zero it
# FIX: INSURE THAT ELEMENTS CAN BE FLOATS HERE:  GET AND ASSIGN SHAPE RATHER THAN COPY? XXX
        self.inputValue = self.variable.copy() * 0.0

        # Assign self.inputState to first inputState in dict
        try:
            self.inputState = list(self.inputStates.values())[0]
        except AttributeError:
            self.inputState = None

    def instantiate_execute_method_parameter_states(self, context=NotImplemented):
        """Call instantiate_mechanism_state_list() to instantiate MechanismParameterStates for subclass' execute method

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

                from Functions.MechanismStates.MechanismState import instantiate_mechanism_state
                from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
                self.executeMethodParameterStates[param_name] = instantiate_mechanism_state(
                                                                                    owner=self,
                                                                                    state_type=MechanismParameterState,
                                                                                    state_name=param_name,
                                                                                    state_spec=param_state_spec,
                                                                                    constraint_values=param_state_spec,
                                                                                    constraint_values_name=param_name,
                                                                                    context=context)

    def instantiate_output_states(self, context=NotImplemented):
        """Call MechanismState.instantiate_mechanism_state_list() to instantiate orderedDict of outputState(s)

        Create OrderedDict of outputState(s) specified in paramsCurrent[kwMechanismInputStates]
        If kwMechanismInputStates is not specified, use self.variable to create a default output state
        When completed:
            - self.outputStates contains an OrderedDict of one or more outputStates
            - self.outputState contains first or only outputState in OrderedDict
            - paramsCurrent[kwMechanismOutputStates] contains the same OrderedDict (of one or more outputStates)
            - each outputState corresponds to an item in the output of the mechanism's execute method (EMO)
            - if there is only one outputState, it is assigned the full value

        (See MechanismState.instantiate_mechanism_state_list() for additional details)

        IMPLEMENTATION NOTE:
            default(s) for self.paramsCurrent[kwMechanismOutputStates] (kwExecuteOutputDefault) is assigned here
            rather than in validate_params, as it requires execute method to have been instantiated first

        :param context:
        :return:
        """
        from Functions.MechanismStates.MechanismState import instantiate_mechanism_state_list
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        self.outputStates = instantiate_mechanism_state_list(owner=self,
                                                            state_list=self.paramsCurrent[kwMechanismOutputStates],
                                                             state_type=MechanismOutputState,
                                                             state_param_identifier=kwMechanismOutputStates,
                                                             constraint_values=self.value,
                                                             constraint_values_name="execute method output",
                                                             context=context)
        # Assign self.outputState to first outputState in dict
        self.outputState = list(self.outputStates.values())[0]

    def add_projection_to_mechanism(self, projection, state, context=NotImplemented):

        from Functions.Projections.Projection import add_projection_to
        add_projection_to(receiver=self, projection_spec=projection, state=state, context=context)

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

        #region VALIDATE INPUT STATE(S) AND RUNTIME PARAMS
        # # MODIFIED 7/26/16 OLD:
        # self.check_args(variable=self.inputState.value,
        # MODIFIED 7/26/16 NEW:
        self.check_args(variable=self.inputValue,
        # MODIFIED 7/26/16 END
                        params=runtime_params,
                        target_set=runtime_params)
        #endregion

        #region UPDATE INPUT STATE(S)
        self.update_input_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region UPDATE PARAMETER STATE(S)
        # #TEST:
        # print ("BEFORE param update:  DDM Drift Rate {}".
        #        format(self.executeMethodParameterStates[kwDDM_DriftRate].value))
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
        """Assign items in self.value to each outputState in outputSates

        Assign each item of self.execute's return value to the value of the corresponding outputState in outputSates
        Use mapping of items to outputStates in self.outputStateValueMapping
        Notes:
        * self.outputStateValueMapping must be implemented by Mechanism subclass (typically in its executeMethod)
        * if len(self.value) == 1, then an absence of self.outputStateValueMapping is forgiven
        * if the executeMethod of a Function is specified only by kwExecuteMethod and returns a value with len > 1
            it MUST also specify kwExecuteMethodOutputStateValueMapping

        """
        if len(self.value) == 1:
            self.outputStates[list(self.outputStates.keys())[0]].value = self.value[0]
        #
        # Assign items in self.value to outputStates using mapping of states to values in self.outputStateValueMapping
        else:
            for state in self.outputStates:
                try:
                    self.outputStates[state].value = self.value[self.outputStateValueMapping[state]]
                except AttributeError:
                    raise MechanismError("{} must implement outputStateValueMapping attribute in executeMethod".
                                         format(self.__class__.__name__))

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
