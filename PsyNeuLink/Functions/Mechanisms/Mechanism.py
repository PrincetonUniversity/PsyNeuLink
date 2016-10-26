# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  MECHANISM MODULE ***********************************************************

"""

Overview
--------

Mechanisms are the core object type in PsyNeuLink.  A mechanism takes an input, transforms it in some way, and provides
it as an output that can be used for some purpose  There are three types of mechanisms that serve different purposes:

* **ProcessingMechanisms** [LINK]
  aggregrate the input they receive from other mechanisms in a process or system, and/or the input to a process or
  system, transform it in some way, and provide the result either as input for other mechanisms and/or the output of a
  process or system.

* **MonitoringMechanisms** [LINK]
  monitor the output of one or more other mechanisms, receive training (target) values, and compare these to generate
  error signals used for learning [LINK].

* **ControlMechanisms** [LINK]
  evaluate the output of one or more other mechanisms, and use this to modify the parameters of those or other
  mechanisms.


COMMENT:
  MOVE TO ProcessingMechanisms overview:
  Different ProcessingMechanisms transform their input in different ways, and some allow this to be customized
  by modifying their ``function`` [LINK].  For example, a ``Transfer`` mechanism can be configured to produce a
  linear, logistic, or exponential transform of its input.
COMMENT

A mechanism is made up of two fundamental components: the function it uses to transform its input; and the states it
uses to represent its input, function parameters, and output

Function
--------

The core of every mechanism is its function, which transforms its input and generates its output.  The function is
specified by the mechanism's ``function`` parameter.  Each type of mechanism specifies one or more functions to use,
and generally these are from the Utility class [LINK] of functions provided by PsyNeuLink.  Functions are specified
in the same form that an object is instantiated in Python (by calling its __init__ method), and thus can be used to
specify its parameters.  For example, for a Transfer mechanism, if the Logistic function is selected, then its gain
and bias parameters can also be specified as shown in the following example::

    my_mechanism = Transfer(function=Logistic(gain=1.0, bias=-4))

While every mechanism type offers a standard set of functions, a custom function can also be specified.  Custom
functions can be any Python function, including an inline (lambda) function, so long as it generates a type of result
that is consistent with the mechanism's type [LINK].

The input to a mechanism's function is contained in the mechanism's ``variable`` attribute, and the result of its
function is contained in the mechanism's ``value`` attribute.

.. note::
   The input to a mechanism is not necessarily the same as the input to its function (i.e., its ``variable`` attribute);
   the mechanism's input is processed by its ``inputState(s)`` before being submitted to its function (see InputStates
   below) [LINK].  Similarly, the result of a mechanism's function (i.e., its ``value`` attribute)  is not necessarily
   the same as the mechanism's output;  the result of the function is processed by the mechanism's ``outputstate(s)``
   which is then assigned to the mechanism's ``outputValue`` attribute (see OutputStates below)[LINK]

States
------

Every mechanism has three types of states:

* **InputStates** [LINK] represent the input(s) to a mechanism.  Generally a mechanism has only one InputState,
  stored in its ``inputState`` attribute.  However some mechs have more than one...

AGGREGATION OF INPUTS
STATE'S FUNCTION (USUALLY COMBINATION FUNCTION)
INPUTSTATE.VARIABLE:  INPUT TO INPUTSTATE
INPUTSTATE.VALUE: OUTPUT OF INPUTSTATE'S FUNCTION,
    INPUT TO MECHANISM'S FUNCTION (MECHANISM.VARIABLE),
    AND == INPUTVALUE??
USUALLY JUST ONE (PRIMARY), BUT CAN BE SEVERAL INPUTSTATES (E.G. COMPARATOR MECHANISM)

* **ParameterStates** [LINK] represent the parameters of a mechanism's function.

* **OutputStates** [LINK] represent the output(s) of a mechainsm.

STATE'S FUNCTION (USUALLY IDENTITY FUNCTION (LINEAR TRANSFER WITH SLOPE = 1 AND INTERCEPT = 0)
OUTPUTSTATE.VARIABLE:  OUTPUT OF MECHANISMS' FUNCTION, INPUT TO OUTPUTSTATE'S FUNCTION
OUTPUTSTATE.VALUE: OUTPUT OF OUTPUTSTATE'S FUNCTION, AND == OUTPUT VALUE OF MECHANISM
USUALLY JUST ONE (PRIMARY), BUT CAN BE SEVERAL OUTPUTSTATES
    (E.G. CONTROL MECHANISM:  ONE FOR EACH PARAMETER CONTROLLED,
     OR FOR DDM THAT CONTAINS DIFFERENT PROPERTIES OF THE OUTPUT)



Role in Processes and Systems
-----------------------------

- DESIGNATION TYPES (in context of a process or system):
        ORIGIN, TERMINAL, SINGLETON, INITIALIZE, INITIALIZE_CYLE, or INTERNAL

ORIGIN GETS MAPPING PROJECTION FROM PROCESSINPUTSTATE

Custom Mechanisms
-----------------




Mechanism specification (from Process):
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
                                + INPUT_STATE_PARAMS:<dict>
                                + PARAMETER_STATE_PARAMS:<dict>
                           [TBI + OUTPUT_STATE_PARAMS:<dict>]
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
"""

from collections import OrderedDict
from inspect import isclass

from PsyNeuLink.Functions.ShellClasses import *
from PsyNeuLink.Globals.Registry import register_category

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
# MODIFIED 9/18/16 NEW:
def mechanism(mech_spec=NotImplemented, params=None, context=None):
# def mechanism(mech_spec=NotImplemented, params=NotImplemented, context=None):
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

    # Called with a keyword
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
        Mechanisms are used as part of a pathway (together with projections) to execute a process
        A mechanism is associated with a name, a set of states, an update_and_execute method, and an execute method:
        - one or more inputStates:
            the value of each represents the aggregated input from its incoming mapping projections, and is used as
            the corresponding item in the variable of the mechanism's execute method
            * Note:
                by default, a Mechanism has only one inputState, assigned to <mechanism>.inputState;  however:
                if the specification of the Mechanism's variable is a list of arrays (lists), or
                if params[kwinputStates] is a list (of names) or a specification dict (of InputState specs),
                    <mechanism>.inputStates (note plural) is created and contains a dict of inputStates,
                    the first of which points to <mechanism>.inputState (note singular)
        - a set of parameters, each of which must be (or resolve to) a reference to a ParameterState
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
            - the nth instance created will be named using the following format: functionType-n
        - by calling mechanism(name, params):
            - if name is the name of a mechanism class in the MechanismRegistry dictionary, it will be instantiated;
                otherwise, the default mechanism will be instantiated with that name
            - params (optional) must be a dictionary with parameter values relevant to the class invoked (see below)
        - by calling a subclass directly (e.g., DDM(name, params)); the name will be assigned to the instance
        - any parameters included in the initialization will be used as defaults for all calls to the mechanism
        - whenever a new subclass is instantiated (either through import or by one of the methods above),
            it is registered in the MechanismRegistry, which keeps a record of the subClass and instanceCount

    Initialization arguments:
        - variable:  establishes type of variable for the execute method, and initializes it (default: ??)
        - params (dict): (see _validate_params below and State.instantiate_state() for details)
            + kwInputState (value, list, dict):
                if param is absent:
                   a default InputState will be instantiated using variable of mechanism's execute method (EMV)
                    it will be placed as the single entry in an OrderedDict
                if param is a single value:
                    it will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is a list:
                    each item will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is an OrderedDict:
                    each entry will (if necessary) be instantiated as a InputState
                in each case, the result will be an OrderedDict of one or more entries:
                    the key for the entry will be the name of the inputState if provided, otherwise
                        kwInputStates-n will used (with n incremented for each entry)
                    the value of the inputState in each entry will be used as the corresponding value of the EMV
                    the dict will be assigned to both self.inputStates and paramsCurrent[kwInputState]
                    self.inputState will be pointed to self.inputStates[0] (the first entry of the dict)
                notes:
                    * if there is only one inputState, but the EMV has more than one item, it is assigned to the
                        the sole inputState, which is assumed to have a multi-item value
                    * if there is more than one inputState, number must match length of EMV, or an exception is raised
                specification of the param value, list item, or dict enrty value can be any of the following,
                    as long as it is compatible with the variable of the mechanism's execute method (EMV):
                    + InputState class: default will be instantiated using EMV as its value
                    + InputState object: its value must be compatible with EMV
                    + Projection subclass ref:
                        default InputState will be instantiated using EMV as its value
                        default projection (for InputState) will be instantiated using EMV as its variable
                            and assigned to InputState
                    + Projection object:
                        InputState will be instantiated using output of projection as its value;
                        this must be compatible with EMV
                    + specification dict:  InputState will be instantiated using EMV as its value;
                        must contain the following entries: (see Initialization arguments for State):
                            + FUNCTION (method)
                            + FUNCTION_PARAMS (dict)
                            + STATE_PROJECTIONS (Projection, specifications dict, or list of either of these)
                    + ParamValueProjection:
                        value will be used as variable to instantiate a default InputState
                        projection will be assigned as projection to InputState
                    + value: will be used as variable to instantiate a default InputState
                * note: inputStates can also be added using State.instantiate_state()
            + FUNCTION:(method):  method used to transform mechanism input to its output;
                this must be implemented by the subclass, or an exception will be raised
                each item in the variable of this method must be compatible with the corresponding InputState
                each item in the output of this method must be compatible  with the corresponding OutputState
                for any parameter of the method that has been assigned a ParameterState,
                    the output of the parameter state's own execute method must be compatible with
                    the value of the parameter with the same name in paramsCurrent[FUNCTION_PARAMS] (EMP)
            + FUNCTION_PARAMS (dict):
                if param is absent, no parameterStates will be created
                if present, each entry will (if necessary) be instantiated as a ParameterState,
                    and the resulting dict will be placed in <mechanism>.parameterStates
                the value of each entry can be any of those below, as long as it resolves to a value that is
                    compatible with param of the same name in <mechanism>.paramsCurrent[FUNCTION_PARAMS] (EMP)
                    + ParameterState class ref: default will be instantiated using param with same name in EMP
                    + ParameterState object: its value must be compatible with param of same name in EMP
                    + Projection subclass ref:
                        default ParameterState will be instantiated using EMP
                        default projection (for ParameterState) will be instantiated using EMP
                            and assigned to ParameterState
                    + Projection object:
                        ParameterState will be instantiated using output of projection as its value;
                        this must be compatible with EMP
                    + specification dict:  ParameterState will be instantiated using EMP as its value;
                        must contain the following entries: (see Instantiation arguments for ParameterState):
                            + FUNCTION (method)
                            + FUNCTION_PARAMS (dict)
                            + STATE_PROJECTIONS (Projection, specifications dict, or list of either of these)
                    + ParamValueProjection tuple:
                        value will be used as variable to instantiate a default ParameterState
                        projection will be assigned as projection to ParameterState
                    + 2-item tuple [convenience notation;  should use ParamValueProjection for clarity]:
                        first item will be used as variable to instantiate a default ParameterState
                        second item will be assigned as projection to ParameterState
                    + value: will be used as variable to instantiate a default ParameterState
            + kwOutputStates (value, list, dict):
                if param is absent:
                    a default OutputState will be instantiated using output of mechanism's execute method (EMO)
                    it will be placed as the single entry in an OrderedDict
                if param is a single value:
                    it will (if necessary) be instantiated and placed as the single entry in an OrderedDict
                if param is a list:
                    each item will (if necessary) be instantiated and placed in an OrderedDict
                if param is an OrderedDict:
                    each entry will (if necessary) be instantiated as a OutputState
                in each case, the result will be an OrderedDict of one or more entries:
                    the key for the entry will be the name of the outputState if provided, otherwise
                        kwOutputStates-n will used (with n incremented for each entry)
                    the value of the outputState in each entry will be assigned to the corresponding item of the EMO
                    the dict will be assigned to both self.outputStates and paramsCurrent[kwOutputStates]
                    self.outputState will be pointed to self.outputStates[0] (the first entry of the dict)
                notes:
                    * if there is only one outputState, but the EMV has more than one item, it is assigned to the
                        the sole outputState, which is assumed to have a multi-item value
                    * if there is more than one outputState, number must match length of EMO, or an exception is raised
                specification of the param value, list item, or dict entry value can be any of the following,
                    as long as it is compatible with (relevant item of) output of the mechanism's execute method (EMO):
                    + OutputState class: default outputState will be instantiated using EMO as its value
                    + OutputState object: its value must be compatible with EMO
                    + specification dict:  OutputState will be instantiated using EMO as its value;
                        must contain the following entries: (see Initialization arguments for State):
                            + FUNCTION (method)
                            + FUNCTION_PARAMS (dict)
                    + str:
                        will be used as name of a default outputState (and key for its entry in self.outputStates)
                        value must match value of the corresponding item of the mechanism's EMO
                    + value:
                        will be used a variable to instantiate a OutputState; value must be compatible with EMO
                * note: inputStates can also be added using State.instantiate_state()
            + MONITORED_OUTPUT_STATES (list): (default: PRIMARY_OUTPUT_STATES)
                specifies the outputStates of the mechanism to be monitored by ControlMechanism of the System(s)
                    to which the Mechanism belongs
                this specification overrides (for this Mechanism) any in the ControlMechanism or System params[]
                this is overridden if None is specified for MONITORED_OUTPUT_STATES in the outputState itself
                each item must be one of the following:
                    + OutputState (object)
                    + OutputState name (str)
                    + (Mechanism or OutputState specification, exponent, weight) (tuple):
                        + mechanism or outputState specification (Mechanism, OutputState, or str):
                            referenceto Mechanism or OutputState object or the name of one
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
            + [TBI: kwMechanismExecutionSequenceTemplate (list of States):
                specifies order in which types of States are executed;  used by self.execute]
        + paramNames (dict)
        + defaultMechanism (str): Currently kwDDM (class reference resolved in __init__.py)

    Class methods:
        - _validate_variable(variable, context)
        - _validate_params(request_set, target_set, context)
        - update_states_and_execute(time_scale, params, context):
            updates input, param values, executes <subclass>.function, returns outputState.value
        - terminate_execute(self, context=None): terminates execution of mechanism (for TimeScale = time_step)
        - adjust(params, context)
            modifies specified mechanism params (by calling Function.assign_defaults)
            returns output
            # parses, validates and assigns validated control signal (called by __init__() and adjust() class methods)

    Instance attributes (implemented in __init__():
        + variable (value): used as input to mechanism's execute method
        + paramsCurrent (dict): current value of all params for instance (created and validated in Functions init)
        + paramInstanceDefaults (dict): defaults for instance (created and validated in Functions init)
        + paramNames (list): list of keys for the params in paramInstanceDefaults
        + inputState (InputState): default MechanismInput object for mechanism
        + inputStates (dict): created if params[kwInputState] specifies  more than one InputState
        + inputValue (value, list or ndarray): value, list or array of values, one for the value of each inputState
        + receivesProcessInput (bool): flags if Mechanism (as first in Pathway) receives Process input projection
        + parameterStates (dict): created if params[FUNCTION_PARAMS] specifies any parameters
        + outputState (OutputState): default OutputState for mechanism
        + outputStates (dict): created if params[kwOutputStates] specifies more than one OutputState
        + value (value, list, or ndarray): output of the Mechanism's execute method;
            Note: currently each item of self.value corresponds to value of corresponding outputState in outputStates
        + outputStateValueMapping (dict): specifies index of each state in outputStates,
            used in _update_output_states to assign the correct item of value to each outputState in outputStates
            Notes:
            * any Function with a function that returns a value with len > 1 MUST implement self.execute
            *    rather than just use the params[FUNCTION] so that outputStateValueMapping can be implemented
            * TBI: if the function of a Function is specified only by params[FUNCTION]
                       (i.e., it does not implement self.execute) and it returns a value with len > 1
                       it MUST also specify kwFunctionOutputStateValueMapping
        + phaseSpec (int or float): time_step(s) on which Mechanism.update() is called (see Process for specification)
        + stateRegistry (Registry): registry containing dicts for each state type (input, output and parameter)
            with instance dicts for the instances of each type and an instance count for each state type
            Note: registering instances of state types with the mechanism (rather than in the StateRegistry)
                  allows the same name to be used for instances of a state type belonging to different mechanisms
                  without adding index suffixes for that name across mechanisms
                  while still indexing multiple uses of the same base name within a mechanism
        + processes (dict):
            entry for each process to which the mechanism belongs; key = process; value = ORIGIN, INTERNAL, OR TERMINAL
            these are use
        + systems (dict):
            entry for each system to which the mechanism belongs; key = system; value = ORIGIN, INTERNAL, OR TERMINAL
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as an arg, default is created by copying Mechanism_BasePreferenceSet

    Instance methods:
        The following method MUST be overridden by an implementation in the subclass:
        - execute:
            - called by update_states_and_execute()
            - must be implemented by Mechanism subclass, or an exception is raised
        - run:
            - calls run() function with mechanism as object
        - _assign_input:
            - called by execute() if call to execute was direct call (i.e., not from process or system) and with input
        - initialize:
            - called by system and process
            - assigns self.value and calls _update_output_states
        - _report_mechanism_execution(input, params, output)

        [TBI: - terminate(context) -
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


    #FIX:  WHEN CALLED BY HIGHER LEVEL OBJECTS DURING INIT (e.g., PROCESS AND SYSTEM), SHOULD USE FULL Mechanism.execute
    # By default, init only the __execute__ method of Mechanism subclass objects when their execute method is called;
    #    that is, DO NOT run the full Mechanism execute process, since some components may not yet be instantiated
    #    (such as outputStates)
    initMethod = INIT__EXECUTE__METHOD_ONLY

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
        MONITORED_OUTPUT_STATES: NotImplemented,
        MONITOR_FOR_LEARNING: NotImplemented
        # TBI - kwMechanismExecutionSequenceTemplate: [
        #     Functions.States.InputState.InputState,
        #     Functions.States.ParameterState.ParameterState,
        #     Functions.States.OutputState.OutputState]
        })

    # def __new__(cls, *args, **kwargs):
    # def __new__(cls, name=NotImplemented, params=NotImplemented, context=None):
    #endregion

    def __init__(self,
                 variable=NotImplemented,
                 params=NotImplemented,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign name, category-level preferences, register mechanism, and enforce category methods

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

        Initialization arguments:
            - input_template (value, InputState or specification dict for one):
                  if value, it will be used as variable (template of self.inputState.value)
                  if State or specification dict, it's value attribute will be used
            - params (dict): dictionary with entries for each param of the mechanism subclass;
                the key for each entry should be the name of the param (used to name its associated projections)
                the value for each entry MUST be one of the following (see Parameters above for details):
                    - ParameterState object
                    - dict: State specifications (see State)
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

        :param input_template: (value)
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

# IMPLEMENT **args (PER State)

        # Register with MechanismRegistry or create one
        if not context is kwValidate:
            register_category(entry=self,
                              base_class=Mechanism_Base,
                              name=name,
                              registry=MechanismRegistry,
                              context=context)

        # # MODIFIED 9/11/16 NEW:
        # Create mechanism's stateRegistry and state type entries
        from PsyNeuLink.Functions.States.State import State_Base
        self.stateRegistry = {}
        # InputState
        from PsyNeuLink.Functions.States.InputState import InputState
        register_category(entry=InputState,
                          base_class=State_Base,
                          registry=self.stateRegistry,
                          context=context)
        # ParameterState
        from PsyNeuLink.Functions.States.ParameterState import ParameterState
        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self.stateRegistry,
                          context=context)
        # OutputState
        from PsyNeuLink.Functions.States.OutputState import OutputState
        register_category(entry=OutputState,
                          base_class=State_Base,
                          registry=self.stateRegistry,
                          context=context)
        # MODIFIED 9/11/16 END

        if not context or isinstance(context, object) or inspect.isclass(context):
            context = kwInit + self.name + kwSeparatorBar + self.__class__.__name__
        else:
            context = context + kwSeparatorBar + kwInit + self.name

        super(Mechanism_Base, self).__init__(variable_default=variable,
                                             param_defaults=params,
                                             prefs=prefs,
                                             name=name,
                                             context=context)

        # FUNCTIONS:

# IMPLEMENTATION NOTE:  REPLACE THIS WITH ABC (ABSTRACT CLASS)
        # Assign class functions
        self.classMethods = {
            kwMechanismExecuteFunction: self.execute,
            kwMechanismAdjustFunction: self.adjust_function,
            kwMechanismTerminateFunction: self.terminate_execute
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
        self.processes = {}
        self.systems = {}

    def _validate_variable(self, variable, context=None):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        # VARIABLE SPECIFICATION:                                        ENCODING:
        # Simple value variable:                                         0 -> [array([0])]
        # Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        # Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        :param variable:
        :param context:
        :return:
        """

        super(Mechanism_Base, self)._validate_variable(variable, context)

        # Force Mechanism variable specification to be a 2D array (to accomodate multiple input states - see above):
        # Note: instantiate_input_states (below) will parse into 1D arrays, one for each input state
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """validate TimeScale, inputState(s), execute method param(s) and outputState(s)

        Call super (Function._validate_params()
        Go through target_set params (populated by Function._validate_params) and validate values for:
            + kwTimeScale:  <TimeScale>
            + kwInputState:
                <MechanismsInputState or Projection object or class,
                specification dict for one, ParamValueProjection tuple, or numeric value(s)>;
                if it is missing or not one of the above types, it is set to self.variable
            + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
                ParameterState or Projection object or class, specification dict for one,
                ParamValueProjection tuple, or numeric value(s);
                if invalid, default (from paramInstanceDefaults or paramClassDefaults) is assigned
            + kwOutputStates:
                <MechanismsOutputState object or class, specification dict, or numeric value(s);
                if it is missing or not one of the above types, it is set to None here;
                    and then to default value of self.value (output of execute method) in instantiate_outputState
                    (since execute method must be instantiated before self.value is known)
                if kwOutputStates is a list or OrderedDict, it is passed along (to instantiate_outputStates)
                if it is a OutputState class ref, object or specification dict, it is placed in a list

        TBI - Generalize to go through all params, reading from each its type (from a registry),
                                   and calling on corresponding subclass to get default values (if param not found)
                                   (as PROJECTION_TYPE and kwProjectionSender are currently handled)

        :param request_set: (dict)
        :param target_set: (dict)
        :param context: (str)
        :return:
        """

        # Perform first-pass validation in Function.__init__():
        # - returns full set of params based on subclass paramClassDefaults
        super(Mechanism, self)._validate_params(request_set,target_set,context)

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
            param_value = params[kwInputStates]

        except KeyError:
            # kwInputStates not specified:
            # - set to None, so that it is set to default (self.variable) in instantiate_inputState
            # - if in VERBOSE mode, warn in instantiate_inputState, where default value is known
            params[kwInputStates] = None

        else:
            # kwInputStates is specified, so validate:
            # If it is a single item or a non-OrderedDict, place in a list (for use here and in instantiate_inputState)
            if not isinstance(param_value, (list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            # Note:
            # * number of inputStates is validated against length of the owner mechanism's execute method variable (EMV)
            #     in instantiate_inputState, where an inputState is assigned to each item (value) of the EMV
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from PsyNeuLink.Functions.States.InputState import InputState
                # If not valid...
                if not ((isclass(item) and (issubclass(item, InputState) or # InputState class ref
                                                issubclass(item, Projection))) or    # Project class ref
                            isinstance(item, InputState) or      # InputState object
                            isinstance(item, dict) or                     # InputState specification dict
                            isinstance(item, ParamValueProjection) or     # ParamValueProjection tuple
                            isinstance(item, str) or                      # Name (to be used as key in inputStates dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):   # value
                    # set to None, so it is set to default (self.variable) in instantiate_inputState
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " InputState, specification dict or value, nor a list of dict of them; "
                              "variable ({4}) of execute method for {5} will be used"
                              " to create a default outputState for {3}".
                              format(i,
                                     kwInputStates,
                                     param_value,
                                     self.__class__.__name__,
                                     self.variable,
                                     self.execute.__self__.name))
                i += 1
            params[kwInputStates] = param_value
        #endregion

        #region VALIDATE EXECUTE METHOD PARAMS
        try:
            function_param_specs = params[FUNCTION_PARAMS]
        except KeyError:
            if self.prefs.verbosePref:
                print("No params specified for {0}".format(self.__class__.__name__))
        else:
            if not (isinstance(function_param_specs, dict)):
                raise MechanismError("{0} in {1} must be a dict of param specifications".
                                     format(FUNCTION_PARAMS, self.__class__.__name__))
            # Validate params
            from PsyNeuLink.Functions.States.ParameterState import ParameterState
            for param_name, param_value in function_param_specs.items():
                try:
                    default_value = self.paramInstanceDefaults[FUNCTION_PARAMS][param_name]
                except KeyError:
                    raise MechanismError("{0} not recognized as a param of execute method for {1}".
                                         format(param_name, self.__class__.__name__))
                if not ((isclass(param_value) and
                             (issubclass(param_value, ParameterState) or
                                  issubclass(param_value, Projection))) or
                        isinstance(param_value, ParameterState) or
                        isinstance(param_value, Projection) or
                        isinstance(param_value, dict) or
                        isinstance(param_value, ParamValueProjection) or
                        iscompatible(param_value, default_value)):
                    params[FUNCTION_PARAMS][param_name] = default_value
                    if self.prefs.verbosePref:
                        print("{0} param ({1}) for execute method {2} of {3} is not a ParameterState, "
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
            param_value = params[kwOutputStates]

        except KeyError:
            # kwOutputStates not specified:
            # - set to None, so that it is set to default (self.value) in instantiate_outputState
            # Notes:
            # * if in VERBOSE mode, warning will be issued in instantiate_outputState, where default value is known
            # * number of outputStates is validated against length of owner mechanism's execute method output (EMO)
            #     in instantiate_outputState, where an outputState is assigned to each item (value) of the EMO
            params[kwOutputStates] = None

        else:
            # kwOutputStates is specified, so validate:
            # If it is a single item or a non-OrderedDict, place in a list (for use here and in instantiate_outputState)
            if not isinstance(param_value, (list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from PsyNeuLink.Functions.States.OutputState import OutputState
                # If not valid...
                if not ((isclass(item) and issubclass(item, OutputState)) or # OutputState class ref
                            isinstance(item, OutputState) or   # OutputState object
                            isinstance(item, dict) or                   # OutputState specification dict
                            isinstance(item, str) or                    # Name (to be used as key in outputStates dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):  # value
                    # set to None, so it is set to default (self.value) in instantiate_outputState
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " OutputState, specification dict or value, nor a list of dict of them; "
                              "output ({4}) of execute method for {5} will be used"
                              " to create a default outputState for {3}".
                              format(i,
                                     kwOutputStates,
                                     param_value,
                                     self.__class__.__name__,
                                     self.value,
                                     self.execute.__self__.name))
                i += 1
            params[kwOutputStates] = param_value

        # MODIFIED 7/13/16 NEW: [MOVED FROM EVCMechanism]
        # FIX: MOVE THIS TO FUNCTION, OR ECHO IN SYSTEM
        #region VALIDATE MONITORED STATES (for use by ControlMechanism)
        # Note: this must be validated after kwOutputStates as it can reference entries in that param
        try:
            # MODIFIED 7/16/16 NEW:
            if not target_set[MONITORED_OUTPUT_STATES] or target_set[MONITORED_OUTPUT_STATES] is NotImplemented:
                pass
            # MODIFIED END
            # It is a MonitoredOutputStatesOption specification
            elif isinstance(target_set[MONITORED_OUTPUT_STATES], MonitoredOutputStatesOption):
                # Put in a list (standard format for processing by instantiate_monitored_output_states)
                target_set[MONITORED_OUTPUT_STATES] = [target_set[MONITORED_OUTPUT_STATES]]
            # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
            else:
                # Validate each item of MONITORED_OUTPUT_STATES
                for item in target_set[MONITORED_OUTPUT_STATES]:
                    self._validate_monitored_state(item, context=context)
                # FIX: PRINT WARNING (IF VERBOSE) IF WEIGHTS or EXPONENTS IS SPECIFIED,
                # FIX:     INDICATING THAT IT WILL BE IGNORED;
                # FIX:     weights AND exponents ARE SPECIFIED IN TUPLES
                # FIX:     WEIGHTS and EXPONENTS ARE VALIDATED IN SystemContro.Mechanisminstantiate_monitored_output_states
                # # Validate WEIGHTS if it is specified
                # try:
                #     num_weights = len(target_set[FUNCTION_PARAMS][WEIGHTS])
                # except KeyError:
                #     # WEIGHTS not specified, so ignore
                #     pass
                # else:
                #     # Insure that number of weights specified in WEIGHTS
                #     #    equals the number of states instantiated from MONITORED_OUTPUT_STATES
                #     num_monitored_states = len(target_set[MONITORED_OUTPUT_STATES])
                #     if not num_weights != num_monitored_states:
                #         raise MechanismError("Number of entries ({0}) in WEIGHTS of kwFunctionParam for EVC "
                #                        "does not match the number of monitored states ({1})".
                #                        format(num_weights, num_monitored_states))
        except KeyError:
            pass
        #endregion
        # MODIFIED END

# FIX: MAKE THIS A CLASS METHOD OR MODULE FUNCTION
# FIX:     SO THAT IT CAN BE CALLED BY System TO VALIDATE IT'S MONITORED_OUTPUT_STATES param

    def _validate_monitored_state(self, state_spec, context=None):
        """Validate specification is a Mechanism or OutputState object or the name of one

        Called by both self._validate_params() and self.add_monitored_state() (in ControlMechanism)
        """
        state_spec_is_OK = False

        if isinstance(state_spec, MonitoredOutputStatesOption):
            state_spec_is_OK = True

        if isinstance(state_spec, tuple):
            if len(state_spec) != 3:
                raise MechanismError("Specification of tuple ({0}) in MONITORED_OUTPUT_STATES for {1} "
                                     "has {2} items;  it should be 3".
                                     format(state_spec, self.name, len(state_spec)))

            if not isinstance(state_spec[1], numbers.Number):
                raise MechanismError("Specification of the exponent ({0}) for MONITORED_OUTPUT_STATES of {1} "
                                     "must be a number".
                                     format(state_spec, self.name, state_spec[0]))

            if not isinstance(state_spec[2], numbers.Number):
                raise MechanismError("Specification of the weight ({0}) for MONITORED_OUTPUT_STATES of {1} "
                                     "must be a number".
                                     format(state_spec, self.name, state_spec[0]))

            # Set state_spec to the output_state item for validation below
            state_spec = state_spec[0]

        from PsyNeuLink.Functions.States.OutputState import OutputState
        if isinstance(state_spec, (Mechanism, OutputState)):
            state_spec_is_OK = True

        if isinstance(state_spec, str):
            if state_spec in self.paramInstanceDefaults[kwOutputStates]:
                state_spec_is_OK = True
        try:
            self.outputStates[state_spec]
        except (KeyError, AttributeError):
            pass
        else:
            state_spec_is_OK = True

        if not state_spec_is_OK:
            raise MechanismError("Specification ({0}) in MONITORED_OUTPUT_STATES for {1} is not "
                                 "a Mechanism or OutputState object or the name of one".
                                 format(state_spec, self.name))
#endregion

    def _validate_inputs(self, inputs=None):
        # Only ProcessingMechanism supports run() method of Function;  ControlMechanism and MonitoringMechanism do not
        raise MechanismError("{} does not support run() method".format(self.__class__.__name__))

    def _instantiate_attributes_before_function(self, context=None):

        self.instantiate_input_states(context=context)

        from PsyNeuLink.Functions.States.ParameterState import instantiate_parameter_states
        instantiate_parameter_states(owner=self, context=context)

        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        # self.instantiate_output_states(context=context)
        from PsyNeuLink.Functions.States.OutputState import instantiate_output_states
        instantiate_output_states(owner=self, context=context)

    def instantiate_input_states(self, context=None):
        """Call State.instantiate_input_states to instantiate orderedDict of inputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override instantiate_input_states
        """

        from PsyNeuLink.Functions.States.InputState import instantiate_input_states
        instantiate_input_states(owner=self, context=context)

    def _add_projection_to_mechanism(self, state, projection, context=None):

        from PsyNeuLink.Functions.Projections.Projection import add_projection_to
        add_projection_to(receiver=self, state=state, projection_spec=projection, context=context)

    def _add_projection_from_mechanism(self, receiver, state, projection, context=None):
        """Add projection to specified state
        """
        from PsyNeuLink.Functions.Projections.Projection import add_projection_from
        add_projection_from(sender=self, state=state, projection_spec=projection, receiver=receiver, context=context)

    def execute(self, input=None, time_scale=TimeScale.TRIAL, runtime_params=None, context=None):
        """Update inputState(s) and param(s), call subclass function, update outputState(s), and assign self.value

        Arguments:
        - time_scale (TimeScale): time scale at which to run subclass execute method
        - params (dict):  params for subclass execute method (overrides its paramInstanceDefaults and paramClassDefaults
        - context (str): should be set to subclass name by call to super from subclass

        Execution sequence:
        - Call self.inputState.execute() for each entry in self.inputStates:
            + execute every self.inputState.receivesFromProjections.[<Projection>.execute()...]
            + aggregate results using self.inputState.params[FUNCTION]()
            + store the result in self.inputState.value
        - Call every self.params[<ParameterState>].execute(); for each:
            + execute self.params[<ParameterState>].receivesFromProjections.[<Projection>.execute()...]
                (usually this is just a single ControlSignal)
            + aggregate results (if > one) using self.params[<ParameterState>].params[FUNCTION]()
            + apply the result to self.params[<ParameterState>].value
        - Call subclass' self.execute(params):
            - use self.inputState.value as its variable,
            - use params[kw<*>] or self.params[<ParameterState>].value for each param of subclass self.execute,
            - apply the output to self.outputState.value
            Note:
            * if execution is occuring as part of initialization, outputState(s) are reset to 0
            * otherwise, they are left in the current state until the next update

        - [TBI: Call self.outputState.execute() (output gating) to update self.outputState.value]

        Returns self.outputState.value and self.outputStates[].value after either one time_step or the full trial
             (set by params[kwMechanismTimeScale)

        :param self:
        :param params: (dict)
        :param context: (optional)
        :rtype outputState.value (list)
        """

        context = context or  kwExecuting + ' ' + append_type_to_name(self)


        # IMPLEMENTATION NOTE: Re-write by calling execute methods according to their order in functionDict:
        #         for func in self.functionDict:
        #             self.functionsDict[func]()

        # Limit init to scope specified by context
        if kwInit in context:
            if kwProcessInit in context or kwSystemInit in context:
                # Run full execute method for init of Process and System
                pass
            # Only call mechanism's __execute__ method for init
            elif self.initMethod is INIT__EXECUTE__METHOD_ONLY:
                return self.__execute__(variable=self.variable,
                                     params=runtime_params,
                                     time_scale=time_scale,
                                     context=context)
            # Only call mechanism's function method for init
            elif self.initMethod is INIT_FUNCTION_METHOD_ONLY:
                return self.function(variable=self.variable,
                                     params=runtime_params,
                                     time_scale=time_scale,
                                     context=context)

        # Direct call to execute mechanism with specified input,
        #    so call subclass __execute__ method with input and runtime_params (if specified), and return
        elif not input is None:
            self._assign_input(input)
            if runtime_params:
                for param_set in runtime_params:
                    if not (INPUT_STATE_PARAMS in param_set or
                            PARAMETER_STATE_PARAMS in param_set or
                            OUTPUT_STATE_PARAMS in param_set):
                        raise MechanismError("{0} is not a valid parameter set for runtime specification".
                                             format(param_set))
            return self.__execute__(variable=input,
                                 params=runtime_params,
                                 time_scale=time_scale,
                                 context=context)

        # Execute

        #region VALIDATE RUNTIME PARAMETER SETS
        # Insure that param set is for a States:
        if self.prefs.paramValidationPref:
            # if runtime_params != NotImplemented:
            if runtime_params:
                for param_set in runtime_params:
                    if not (INPUT_STATE_PARAMS in param_set or
                            PARAMETER_STATE_PARAMS in param_set or
                            OUTPUT_STATE_PARAMS in param_set):
                        raise MechanismError("{0} is not a valid parameter set for runtime specification".
                                             format(param_set))
        #endregion

        #region VALIDATE INPUT STATE(S) AND RUNTIME PARAMS
        self._check_args(variable=self.inputValue,
                        params=runtime_params,
                        target_set=runtime_params)
        #endregion

        #region UPDATE INPUT STATE(S)
        self._update_input_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region UPDATE PARAMETER STATE(S)
        # #TEST:
        # print ("BEFORE param update:  DDM Drift Rate {}".
        #        format(self.parameterStates[DRIFT_RATE].value))
        self._update_parameter_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region CALL SUBCLASS __execute__ method AND ASSIGN RESULT TO self.value
# CONFIRM: VALIDATION METHODS CHECK THE FOLLOWING CONSTRAINT: (AND ADD TO CONSTRAINT DOCUMENTATION):
# DOCUMENT: #OF OUTPUTSTATES MUST MATCH #ITEMS IN OUTPUT OF EXECUTE METHOD **

        self.value = self.__execute__(variable=self.inputValue, time_scale=time_scale, context=context)
        #endregion

        #region UPDATE OUTPUT STATE(S)
        self._update_output_states(time_scale=time_scale, context=context)
        #endregion

        #region TBI
        # # Call outputState.execute
        # #    updates outState.value, based on any projections (e.g., gating) it may get
        # self.inputState.execute()
        #endregion

        #region REPORT EXECUTION
        if self.prefs.reportOutputPref and context and kwExecuting in context:
            self._report_mechanism_execution(self.inputValue, self.user_params, self.outputState.value)

        #endregion

        #region RE-SET STATE_VALUES AFTER INITIALIZATION
        # If this is (the end of) an initialization run, restore state values to initial condition
        if '_init_' in context:
            for state in self.inputStates:
                self.inputStates[state].value = self.inputStates[state].variable
            for state in self.parameterStates:
                self.parameterStates[state].value =  self.parameterStates[state].baseValue
            for state in self.outputStates:
                # Zero outputStates in case of recurrence:
                #    don't want any non-zero values as a residuum of initialization runs to be
                #    transmittted back via recurrent projections as initial inputs
# FIX: IMPLEMENT zero_all_values METHOD
                self.outputStates[state].value = self.outputStates[state].value * 0.0
        #endregion

        #endregion

        return self.value

    def run(self,
            inputs,
            num_executions=None,
            call_before_execution=None,
            call_after_execution=None,
            time_scale=None):
        """Run a sequence of executions

        Call execute method for each in a sequence of executions specified by the ``inputs`` argument.  See Run [LINK]
        for additional details of formatting input specifications)

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_input_value
            input for each execution of mechanism (see ``run`` function [LINK] for detailed
            description of formatting requirements and options).

        call_before_execution : Function : default= ``None``
            called before each execution of the mechanism.

        call_after_execution : Function : default= ``None``
            called after each execution of the mechanism.

        time_scale : TimeScale :  default TimeScale.TRIAL
            determines whether mechanisms are executed for a single time step or a trial

        Returns
        -------

        <mechanism>.results : List[outputState.value]
            list of the values of the outputStates for each execution of the mechanism

        """
        from PsyNeuLink.Globals.Run import run
        return run(self,
                   inputs=inputs,
                   num_executions=num_executions,
                   call_before_trial=call_before_execution,
                   call_after_trial=call_after_execution,
                   time_scale=time_scale)

    def _assign_input(self, input):

        input = np.atleast_2d(input)
        num_inputs = np.size(input,0)
        num_input_states = len(self.inputStates)
        if num_inputs != num_input_states:
            # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
            num_inputs = np.size(input)
            if isinstance(input, np.ndarray) and input.dtype is np.dtype('O') and num_inputs == num_input_states:
                pass
            else:
                raise SystemError("Number of inputs ({0}) to {1} does not match "
                                  "its number of inputStates ({2})".
                                  format(num_inputs, self.name,  num_input_states ))
        for i in range(num_input_states):
            input_state = list(self.inputStates.values())[i]
            # input_item = np.ndarray(input[i])
            input_item = input[i]
            if len(input_state.variable) == len(input_item):
                input_state.value = input_item
            else:
                raise MechanismError("Length ({}) of input ({}) does not match "
                                     "required length ({}) for input to {} of {}".
                                     format(len(input_item),
                                            input[i],
                                            len(input_state.variable),
                                            input_state.name,
                                            append_type_to_name(self)))

    def _update_input_states(self, runtime_params=NotImplemented, time_scale=None, context=None):
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

    def _update_parameter_states(self, runtime_params=NotImplemented, time_scale=None, context=None):
        for state_name, state in self.parameterStates.items():
            state.update(params=runtime_params, time_scale=time_scale, context=context)

    def _update_output_states(self, time_scale=None, context=None):
        """Assign items in self.value to each outputState in outputSates

        Assign each item of self.execute's return value to the value of the corresponding outputState in outputSates
        Use mapping of items to outputStates in self.outputStateValueMapping
        Notes:
        * self.outputStateValueMapping must be implemented by Mechanism subclass (typically in its function)
        * if len(self.value) == 1, (i.e., there is only one value), absence of self.outputStateValueMapping is forgiven
        * if the function of a Function is specified only by FUNCTION and returns a value with len > 1
            it MUST also specify kwFunctionOutputStateValueMapping

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
                    raise MechanismError("{} must implement outputStateValueMapping attribute in function".
                                         format(self.__class__.__name__))

    def initialize(self, value):
        if self.paramValidationPref:
            if not iscompatible(value, self.value):
                raise MechanismError("Initialization value ({}) is not compatiable with value of {}".
                                     format(value, append_type_to_name(self)))
        self.value = value
        self._update_output_states()

    def __execute__(self,
                    variable=NotImplemented,
                    params=NotImplemented,
                    time_scale=None,
                    context=None):
        return self.function(variable=variable, params=params, time_scale=time_scale, context=context)

    def _report_mechanism_execution(self, input=None, params=None, output=None):

        if input is None:
            input = self.inputValue
        if output is None:
            output = self.outputState.value
        params = params or self.user_params

        import re
        if 'mechanism' in self.name or 'Mechanism' in self.name:
            mechanism_string = ' '
        else:
            mechanism_string = ' mechanism'
        print ("\n\'{}\'{} executed:\n- input:  {}".
               format(self.name, mechanism_string, input.__str__().strip("[]")))
        if params:
            print("- params:")
            for param_name, param_value in params.items():
                # No need to report these here, as they will be reported for the function itself below
                if param_name is FUNCTION_PARAMS:
                    continue
                param_is_function = False
                if isinstance(param_value, Function):
                    param = param_value.__self__.__name__
                    param_is_function = True
                elif isinstance(param_value, type(Function)):
                    param = param_value.__name__
                    param_is_function = True
                else:
                    param = param_value
                print ("\t{}: {}".format(param_name, str(param).__str__().strip("[]")))
                if param_is_function:
                    for fct_param_name, fct_param_value in self.function_object.user_params.items():
                        print ("\t\t{}: {}".format(fct_param_name, str(fct_param_value).__str__().strip("[]")))
        print("- output: {}".
              format(re.sub('[\[,\],\n]','',str(output))))

    def adjust_function(self, params, context=None):
        """Modify control_signal_allocations while process is executing

        called by process.adjust()
        returns output after either one time_step or full trial (determined by current setting of time_scale)

        :param self:
        :param params: (dict)
        :param context: (optional)
        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """

        self.assign_defaults(self.inputState, params)
# IMPLEMENTATION NOTE: *** SHOULD THIS UPDATE AFFECTED PARAM(S) BY CALLING self._update_parameter_states??
        return self.outputState.value

    def terminate_execute(self, context=None):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        if context==NotImplemented:
            raise MechanismError("terminate execute method not implemented by mechanism sublcass")

    def get_mechanism_param_values(self):
        """Return dict with current value of each ParameterState in paramsCurrent
        :return: (dict)
        """
        from PsyNeuLink.Functions.States.ParameterState import ParameterState
        return dict((param, value.value) for param, value in self.paramsCurrent.items()
                    if isinstance(value, ParameterState) )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):
        self._value = assignment

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

def is_mechanism_spec(spec):
    """Evaluate whether spec is a valid Mechanism specification

    Return true if spec is any of the following:
    + Mechanism class
    + Mechanism object:
    Otherwise, return False

    Returns: (bool)
    """
    if inspect.isclass(spec) and issubclass(spec, Mechanism):
        return True
    if isinstance(spec, Mechanism):
        return True
    return False


MechanismTuple = namedtuple('MechanismTuple', 'mechanism, params, phase')


from collections import UserList, Iterable
class MechanismList(UserList):
    """Provides access to items and their attributes in a list of mech_tuples for an owner

    The mech_tuples in the list must be MechanismTuples;  that is of the form:
    (mechanism object, runtime_params dict, phaseSpec int)

    Attributes
    ----------
    mechanisms : list of Mechanism objects

    names : list of strings
        each item is a mechanism.name

    values : list of values
        each item is a mechanism.value

    outputStateNames : list of strings
        each item is an outputState.name

    outputStateValues : list of values
        each item is an outputState.value
    """

    def __init__(self, owner, tuples_list:list):
        super().__init__()
        self.mech_tuples = tuples_list
        self.owner = owner
        for item in tuples_list:
            if not isinstance(item, MechanismTuple):
                raise MechanismError("The following item in the tuples_list arg of MechanismList()"
                                     " is not a MechanismTuple: {}".format(item))
        self.process_tuples = tuples_list

    def __getitem__(self, item):
        """Return specified mechanism in MechanismList
        """
        # return list(self.mech_tuples[item])[MECHANISM]
        return self.mech_tuples[item].mechanism

    def __setitem__(self, key, value):
        raise ("MyList is read only ")

    def __len__(self):
        return (len(self.mech_tuples))

    def get_tuple_for_mech(self, mech):
        """Return first mechanism tuple containing specified mechanism from the list of mech_tuples
        """
        if list(item.mechanism for item in self.mech_tuples).count(mech):
            if self.owner.verbosePref:
                print("PROGRAM ERROR:  {} found in more than one mech_tuple in {} in {}".
                      format(append_type_to_name(mech), self.__class__.__name__, self.owner.name))
        return next((mech_tuple for mech_tuple in self.mech_tuples if mech_tuple.mechanism is mech), None)

    @property
    def mechanisms(self):
        """Return list of all mechanisms in MechanismList
        """
        return list(self)

    @property
    def names(self):
        """Return names of all mechanisms in MechanismList
        """
        return list(item.name for item in self.mechanisms)

    @property
    def values(self):
        """Return values of all mechanisms in MechanismList
        """
        return list(item.value for item in self.mechanisms)

    @property
    def outputStateNames(self):
        """Return names of all outputStates for all mechanisms in MechanismList
        """
        names = []
        for item in self.mechanisms:
            for output_state in item.outputStates:
                names.append(output_state)
        return names

    @property
    def outputStateValues(self):
        """Return values of outputStates for all mechanisms in MechanismList
        """
        values = []
        for item in self.mechanisms:
            for output_state_name, output_state in list(item.outputStates.items()):
                values.append(output_state.value)
        return values
