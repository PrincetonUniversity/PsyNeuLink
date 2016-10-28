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

* :doc:`ProcessingMechanism`
  aggregrate the input they receive from other mechanisms in a process or system, and/or the input to a process or
  system, transform it in some way, and provide the result either as input for other mechanisms and/or the output of a
  process or system.

* :doc:`MonitoringMechanism`
  monitor the output of one or more other mechanisms, receive training (target) values, and compare these to generate
  error signals used for learning (see :doc:`Learning`).

* :doc:`ControlMechanism`
  evaluate the output of one or more other mechanisms, and use this to modify the parameters of those or other
  mechanisms.


COMMENT:
  MOVE TO ProcessingMechanisms overview:
  Different ProcessingMechanisms transform their input in different ways, and some allow this to be customized
  by modifying their ``function`` parameter.  For example, a ``Transfer`` mechanism can be configured to produce a
  linear, logistic, or exponential transform of its input.
COMMENT

A mechanism is made up of two fundamental components: the function it uses to transform its input; and the states it
uses to represent its input, function parameters, and output

.. _Mechanism_Creating_A_Mechanism:

Creating a Mechanism
--------------------

Mechanisms can be created in a number of ways.  The simplest is to use the standard Python method of calling the
subclass for the desired type of mechanism.  In addition, PsyNeuLink provides a  ``mechanism`` [LINK] "factory"  method
that can be used to instantiate a specified type of mechanism or a  default mechanism (see [LINK]).  Where other
objects required specification of a mechanism (e.g., the ``pathway`` attribute of a process), this can be done
in either of the ways just mentioned, or in any of the following ways:

  * name of an **existing mechanism**

  * name of a **mechanism type** (class)

  * **specification dictionary** -- this can contain an entry specifying the type of mechanism, and/or entries specifiying
    the value of parameters used to instantiate it.  These should take the following form:

      * :keyword:`MECHANISM_TYPE`: <name of a mechanism type>

          if this entry is absent, a default mechahnism will be creaeted (see [LINK]: Mechanism_Base.defaultMechanism)

      * <name of argument>:<value>

          this can contain any of the standard arguments for instantiating a mechanism (see arguments [LINK] below) or
          those specific to a particular type of mechanism (see documentation for subclass).  Note that parameter values
          in the specification dict will be used to instantiate the mechanism.  These can be overridden during execution
          by specifying runtime parameters, either when calling the ``execute`` method for the mechanism, or where it is
          specified in the ``pathway`` of a process (see [LINK] to processs pathway and to execute below).

COMMENT:
    PUT EXAMPLE HERE
COMMENT

Function
--------

The core of every mechanism is its function, which transforms its input and generates its output.  The function is
specified by the mechanism's ``function`` parameter.  Each type of mechanism specifies one or more functions to use,
and generally these are from the :doc:`UtilityFunction` class provided by PsyNeuLink.  Functions are specified
in the same form that an object is instantiated in Python (by calling its __init__ method), and thus can be used to
specify its parameters.  For example, for a Transfer mechanism, if the Logistic function is selected, then its gain
and bias parameters can also be specified as shown in the following example::

    my_mechanism = Transfer(function=Logistic(gain=1.0, bias=-4))

While every mechanism type offers a standard set of functions, a custom function can also be specified.  Custom
functions can be any Python function, including an inline (lambda) function, so long as it generates a type of result
that is consistent with the mechanism's type (see :doc:`Function`).

The input to a mechanism's function is contained in the mechanism's ``variable`` attribute, and the result of its
function is contained in the mechanism's ``value`` attribute.

.. note::
   The input to a mechanism is not necessarily the same as the input to its function (i.e., its ``variable`` attribute);
   the mechanism's input is processed by its ``inputState(s)`` before being submitted to its function
   (see :ref:`InputStates`).  Similarly, the result of a mechanism's function (i.e., its ``value`` attribute)  is not
   necessarily the same as the mechanism's output;  the result of the function is processed by the mechanism's
   ``outputstate(s)`` which is then assigned to the mechanism's ``outputValue`` attribute (see :ref:`OutputStates`)

States
------

Every mechanism has three types of states (shown schematically in the figure below):

.. figure:: _static/Mechanism_states_fig.*
   :alt: Mechanism States
   :scale: 75 %
   :align: center

   Schematic of a mechanism showing its three types of states (input, parameter and output).

.. _InputStates:

InputStates
~~~~~~~~~~~

These represent the input(s) to a mechanism. A mechanism usually has only one InputState,
stored in its ``inputState`` attribute.  However some mechanisms have more than one.  For example, Comparator
mechanisms have one inputState for their ``sample`` and another for their ``target`` input.  If a mechanism has
more than one inputState, they are stored in an OrderedDict in the mechanisms ``inputStates`` attribute;  the key of
each entry is the name of the inputState and its value is the inputState itself.  If a mechanism has multiple
inputStates, the first -- designated its *primary* inputState -- is also assigned to its ``inputState`` attribute.

Each inputState of a mechanism can
receive one or more projections from other mechanisms, however all must provide values that share the same format
(i.e., number and type of elements).  A list of projections received by an inputState is stored in its
``receivesFromProjections`` attribute.  InputStates, like every other object type in PsyNeuLnk, have a
``function`` parameter.  An inputState's function performs a Hadamard (i.e., elementwise) aggregation of the
inputs it receives from its projections.  The default function is ``LinearCombination`` which simply sums the values
and assigns the result to the inputState's ``value`` attribute.  A custom function can be assigned to an inputState
(e.g., to perform a Hadamard product, or to handle non-numeric values in some way), so long as it generates an output
that is compatible with its inputs the value for that inputState expected by the mechanism's function.  The value
attributes for all a mechanism's inputStates are concatenated into a 2d np.array and assigned to the mechanism's
``variable`` attribute, which serves as the input to the mechanism's function.

COMMENT:
  • Move some of above to States module (or individual state types), and condense??
  • Define ``inputValue`` attribute??
COMMENT

.. _ParameterStates:

ParameterStates
~~~~~~~~~~~~~~~
These represent the parameters of a mechanism's function.  PsyNeuLink assigns one parameterState for each parameter
of the function (which correspond to the arguments in the call to instantiate it; i.e., its ``__init__`` method).
Like other states, parameterStates can receive projections. Typically these are from :doc:`ControlSignal` projections
from a :doc:`ControlMechanism` that is used to modify the function's parameter value in response to the outcome(s)
of processing.  The parameter of a function can also be modified by a specification of runtime parameters with the
mechanism.  The figure below shows how these factors are combined by the parameterState to determine the parameter
of a function.

    **Role of ParameterStates in Controlling the Parameter Value of a Function**

    .. figure:: _static/ParameterState_fig.*
       :alt: ParameterState
       :scale: 75 %

       ..

       +--------------+------------------------------------------------------------------+
       | Component    | Impact on Parameter Value                                        |
       +==============+==================================================================+
       | Brown (A)    | baseValue of drift rate parameter of DDM function                |
       +--------------+------------------------------------------------------------------+
       | Purple (B)   | runtime specification of drift rate parameter                    |
       +--------------+------------------------------------------------------------------+
       | Red (C)      | runtime parameter influences controlSignal-modulated baseValue   |
       +--------------+------------------------------------------------------------------+
       | Green (D)    | combined controlSignals modulate baseValue                       |
       +--------------+------------------------------------------------------------------+
       | Blue (E)     | parameterState function combines controlSignals                  |
       +--------------+------------------------------------------------------------------+

.. _OutputStates:

OutputStates
~~~~~~~~~~~~

These represent the output(s) of a mechanism. A mechanism can have several outputStates.  Similar to inputStates, the
*primary* (first or only) outputState is assigned to the mechanism's ``outputState`` attribute, while all of its
outputStates (including the primary one) are stored in an OrderedDict in its ``outputStates`` attribute;  the
key for each entry is the name of an outputState, and the value is the outputState itself.  Usually the function
of the primary outputState transfers the result of the mechanism's function to the primary outputState's ``value``
attribute (i.e., its function is the Linear function with slope=1 and intercept=0).  Other outputStates may use
other functions to transform the result of the mechanism's function in various ways (e.g., generate its mean,
variance, etc.), the results of which are stored in each outputState's ``value`` attribute.  OutputStates may also
be used for other purposes.  For example, ControlMechanisms can have multiple outputStates, one for each parameter
controlled.  The ``value`` of each outputState can serve as a sender for projections, to transmit its value to other
mechahnisms and/or the ouput of a process or system.  The ``value`` attributes of all of a mechanism's outputStates
are concatenated into a 2d np.array and assigned to the mechanism's ``outputValue`` attribute.


Execution
---------

A mechanism can be executed using its ``execute`` or ``run`` methods.  This can be useful in testing a mechanism
and/or debugging.  However, more typically, mechanisms are executed as part of a process or system (see Process
:ref:`Process_Execution` and System :ref:`System_Execution` for more details).  For either of these, the mechanism must
be included in the ``pathway`` of a process.  There, it can be specified on its own, or as the first item of a tuple
that also has an optional set of runtime parameters (see below), and/or a phase specification for use when executed
in a system (see System :ref:`System_Phase` for an explanation of phases; and see Process :ref:`Process_Mechanisms`
for additional details about specifying a mechanism in a process ``pathway``).

.. note::
   Mechanisms cannot be specified directly in a system.  They must be specified in the ``pathway`` of a process,
   and then that process must be included in the ``processes`` of a system.

.. _Mechanism_Runtime_parameters:

Runtime Parameters
~~~~~~~~~~~~~~~~~~

The parameters of a mechanism are usually specified when the mechanism is created.  However, these can be overridden
when it executed.  This can be done by using the ``runtime_param`` argument of its ``execute`` method, or by specifying
the runtime parameters in a tuple with the mechanism in the ``pathway`` of a process (see :ref:`Process_Mechanisms`).
In either case, runtime parameters  are specified using a dictionary that contains one or more entries, each of which
contains a sub-dictionary corresponding to the mechanism's states (inputStates, parameterStates and/or outputStates);
those dictoinaries, in turn, contain entries for the values of the runtime parameters of the state, its function,
or projections (see the ``runtime_params`` argument of the ``execute`` method below for more details).

Role in Processes and Systems
-----------------------------

Mechanisms that are part of a process and/or system are assigned designations that indicate the role they play.  These
are stored in the mechanism's ``processes`` and ``systems`` attributes, respectively (see Process
:ref:`Process_Mechanisms` and System :ref:`System_Mechanisms` for designation labels and their meanings).
Any mechanism designated as :keyword:`ORIGIN` receives a projection to its primary inputState from the process(es)
to which it belongs.  Accordingly, when the process (or system of which the process is a part) is executed, those
mechainsms receive the input provided to the process (or system).  Note that a mechanism can be the :keyword:`ORIGIN`
of a process but not of a system to which that process belongs (see the note under System :ref:`System_Mechanisms` for
further explanation).  The output value of any mechanism designated as :keyword:`TERMINAL` is assigned to the output
of any process or system to which it belongs.


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


def mechanism(mech_spec=None, params=None, context=None):
    """Factory method for Mechanism; returns the type of mechanism specified or a default mechanism.

    If called with no arguments, returns the default mechanism ([LINK for default]).

    Arguments
    ---------

    mech_spec : Optional[Mechanism subclass, str, or dict]
        if it is :keyword:`None`, returns the default mechanism ([LINK for default]).
        If it is the name of a Mechanism subclass, a default instance of that subclass is returned.
        If it is the name of a Mechanism subclass registered in the ``MechanismRegistry``,
        an instance of a default mechanism for that class is returned;
        otherwise, the string is used to name an instance of the default mechanism.
        If it is a dict, it must be a mechanism specification dict (see :ref:`Mechanism_Creating_A_Mechanism`).
        Note: if a name is not specified, the nth instance created will be named by using the mechanism's
        ``functionType`` attribute as the base and adding an indexed suffix:  functionType-n.

    params : Optional[Dict[param keyword, param value]]
        passed to the relevant subclass to instantiate the mechanism (see ``params`` parameter of
        :any:`Mechanism_Base` below for details of specification).

        FORMAT AND ADD TO ARGUMENTS, WITH NOTE THAT THIS APPLIES TO ALL SUBCLASSES:
            DICT SPECIFICATION
            - one or more inputStates:
                * Note:
                    by default, a Mechanism has only one inputState, assigned to <mechanism>.inputState;  however:
                    if the specification of the Mechanism's variable is a list of arrays (lists), or
                    if params[INPUT_STATES] is a list (of names) or a specification dict (of InputState specs),
                        <mechanism>.inputStates (note plural) is created and contains a dict of inputStates,
                        the first of which points to <mechanism>.inputState (note singular)
            - a set of parameters, each of which must be (or resolve to) a reference to a ParameterState
                these determine the operation of the mechanism's function
            - one or more outputStates:
                the variable of each receives the corresponding item in the output of the mechanism's function
                the value of each is passed to corresponding mapping projections for which the mechanism is a sender
                * Notes:
                    by default, a Mechanism has only one outputState, assigned to <mechanism>.outputState;  however:
                      if params[kwOutputStates] is a list (of names) or specification dict (of MechanismOuput State specs),
                      <mechanism>.outputStates (note plural) is created and contains a dict of outputStates,
                      the first of which points to <mechanism>.outputState (note singular)
                [TBI * each outputState maintains a list of projections for which it serves as the sender]

            - an update_states_and_execute method:
                coordinates updating of inputs, params and execution of mechanism's execute method (implemented by subclass)

           (see _validate_params below and State.instantiate_state() for details)
            + INPUT_STATES (value, list, dict):
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
                    - projection: Projection object, Projection specifications dict, or list of either)
                        + Projection subclass ref:
                            default ParameterState will be instantiated using EMP
                            default projection (for ParameterState) will be instantiated using EMP
                                and assigned to ParameterState
                        + Projection object:
                            ParameterState will be instantiated using output of projection as its value;
                            this must be compatible with EMP
                        + Projection specification dict
                        + List[any of the above]
                    + State specification dict:  ParameterState will be instantiated using EMP as its value;
                        must contain the following entries: (see Instantiation arguments for ParameterState):
                            + FUNCTION (method)
                            + FUNCTION_PARAMS (dict)
                            + STATE_PROJECTIONS (Projection, specifications dict, or list of either of these)
                    + ParamValueProjection tuple:
                        value will be used as variable to instantiate a default ParameterState
                        projection will be assigned as projection to ParameterState
                    + 2-item tuple : (value, projectionType)
                        [convenience notation;  should use ParamValueProjection for clarity]:
                        first item will be used as variable to instantiate a default ParameterState
                        second item will be assigned as projection to ParameterState
                    + value : list of numbers (no projections will be assigned)
                        will be used as variable to instantiate a default ParameterState

            COPIED FROM __init__  (RELABELED WITH PARAMETER_STATES); ADD TO ABOVE??
            + PARAMETER_STATES (dict): dictionary with entries for each param of the mechanism subclass;
                the key for each entry should be the name of the param (used to name its associated projections)
                the value for each entry MUST be one of the following (see Parameters above for details):
                    - ParameterState object
                    - dict: State specifications (see State)
                    - projection: Projection object, Projection specifications dict, or list of either)
                    - tuple: (value, projectionType)
                    - value: list of numbers (no projections will be assigned)

            + OUTPUT_STATES (value, list, dict):
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
    COMMENT

    context : str
        if it is the keyword :keyword:`VALIDATE`, returns :keyword:`True` if specification would return a valid
        subclass object; otherwise returns :keyword:`False`.

    Returns
    -------

    Instance of specified Mechanism subclass or None : Mechanism
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
    """Abstract class for Mechanism

    .. note::
       Mechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the :class:`mechanism` factory method (see it for description of parameters),
       or by calling the desired subclass.

    COMMENT:

        Arguments:
            - variable : value, InputState or specification dict for one
                      if value, it will be used as variable (template of self.inputState.value)
                      if State or specification dict, it's value attribute will be used
            - params : dict
                Dictionary with entries for each param of the mechanism subclass;
                the key for each entry should be the name of the param (used to name its associated projections)
                the value for each entry MUST be one of the following (see Parameters above for details):
                    - ParameterState object
                    - dict: State specifications (see State)
                    - projection: Projection object, Projection specifications dict, or list of either)
                    - tuple: (value, projectionType)
                    - value: list of numbers (no projections will be assigned)
            - name : str
                Mechanisms can be named explicitly (using the name='<name>' argument), which is assigned to self.name.
                If the argument is omitted, it will be assigned the subclass name with a hyphenated,
                indexed suffix ('subclass.name-n')
            - prefs : PreferenceSet or specification dict : Mechanism.classPreferences
                preference set for the mechanism; specified in prefs argument or by ``classPreferences`` is
                defined in __init__.py (see Description under PreferenceSet for details).[LINK]
                if not specified as an arg, default is created by copying Mechanism_BasePreferenceSet
            - context : str
                must be name of subclass;  otherwise raises an exception for direct call

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

        MechanismRegistry:
            All Mechanisms are registered in MechanismRegistry, which maintains a dict for each subclass,
              a count for all instances of that type, and a dictionary of those instances

    COMMENT

    Attributes
    ----------

    variable : value, List[value] or ndarray : default ``variableInstanceDefault``
        value used as input to the mechanism's ``function``.  When specified in the call to create an instance
        (i.e., the mechanism's __init__ method), it is used as a template to define the format of the function's input
        (length and type of elements), and the default value for the instance.
        Converted internally to a 2d np.array.

    inputValue : 2d np.array : default ``variableInstanceDefault``
        synonym for ``variable``; contains one value for the variable of each inputState of the mechanism.

    function_params : Dict[str, value]
        contains one entry for each parameter of the mechanism's function.
        The key of each entry is the name of a function parameter, and the value its value.

    _receivesProcessInput (bool): flags if Mechanism (as first in Pathway) receives Process input projection

    inputState : InputState : default default InputState
        primary inputState for the mechanism;  same as first entry in ``inputStates`` attribute.

    inputStates : OrderedDict[str, InputState]
        contains a dictionary of the mechanism's inputStates.
        The key of each entry is the name of the inputState, and its value is the inputState.
        There is always at least one entry, which contains the primary inputState
        (i.e., the one in the ``inputState`` attribute).

    parameterStates : OrderedDict[str, ParameterState]
        contains a dictionary of parameterStates, one for each parameater of the mechanism's function.
        The key of each entry is the name of the parameterState, and its value is the parameterState.
        Note: mechanism's function parameters are listed in the the ``function_params`` attribute).

    outputState : OutputState : default default OutputState
        primary outputState for the mechanism;  same as first entry in ``outputStates`` attribute.

    outputStates : OrderedDict[str, InputState]
        contains a dictionary of the mechanism's outputStates.
        the key of each entry is the name of an outputState, and its value is the outputState.
        There is always at least one entry, which contains the primary outputState
        (i.e., the one in the ``outputState`` attribute).

    value : 2d np.array : default None
        output of the mechanism's function;
        Note: this is not necessarily equal to the ``outputValue`` attribute;  it is :keyword:`None` until
        the mechanism has been at least once executed;

    outputValue : List[value] : default mechanism.function(variableInstanceDefault)
        list of values of the mechanism's outputStates

    _outputStateValueMapping : Dict[str, int]:
        contains the mappings of outputStates to their indices in the outputValue list
        The key of each entry is the name of an outputState,
        and the value is its position in the ``outputStates`` OrderedDict.
        Used in ``_update_output_states`` to assign the value of each outputState to the correct item of
        the mechanism's ``value`` attribute.
        COMMENT:
            Any mechanism with a function that returns a value with more than one item (i.e., len > 1) MUST implement
                self.execute rather than just use the params[FUNCTION].  This is so that _outputStateValueMapping
                can be implemented
            TBI: if the function of a mechanism is specified only by params[FUNCTION]
                (i.e., it does not implement self.execute) and it returns a value with len > 1
                it MUST also specify kwFunctionOutputStateValueMapping
        COMMENT

    phaseSpec : int or float :  default 0
        specifies the time_step(s) on which the mechanism is executed as part of a system
        (see Process for specification [LINK], and System for how phases are used [LINK])

    _stateRegistry : Registry
        registry containing dicts for each state type (InputState, OutputState and ParameterState)
        with instance dicts for the instances of each type and an instance count for each state type.
        Note: registering instances of state types with the mechanism (rather than in the StateRegistry)
              allows the same name to be used for instances of a state type belonging to different mechanisms
              without adding index suffixes for that name across mechanisms
              while still indexing multiple uses of the same base name within a mechanism

    processes : Dict[Process, str]:
        contains a dictionary of the processes to which the mechanism belongs, and its designation in each.
        The key of each entry is a process to which the mechanism belongs, and its value the mechanism's designation
        in that process (see Process :ref:`Process_Mechanisms` for designations and their meanings).

    systems : Dict[System, str]:
        contains a dictionary of the systems to which the mechanism belongs, and its designation in each.
        The key of each entry is a system to which the mechanism belongs, and its value the mechanism's designation
        in that system (see System :ref:`System_Mechanisms` for designations and their meanings).

    timeScale : TimeScale : default TimeScale.TRIAL
        determines the default TimeScale value used by the mechanism when executed.

    name : str : default Process-[index]
        name of the mechanism; specified in name argument or assigned by MechanismRegistry
        (see Registry module for conventions used in naming, including for default and duplicate names).[LINK]
    COMMENT:
        ??name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
    COMMENT

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        preference set for the mechanism; specified in prefs argument or by ``classPreferences`` is
        defined in __init__.py (see Description under PreferenceSet for details).[LINK]
    COMMENT:
        ???prefs (PreferenceSet): if not specified as an arg, default is created by copying Mechanism_BasePreferenceSet
    COMMENT


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

        NOTES:
        * Since Mechanism is a subclass of Function, it calls super.__init__
            to validate variable_default and param_defaults, and assign params to paramInstanceDefaults;
            it uses kwInputState as the variable_default
        * registers mechanism with MechanismRegistry

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

        # Create mechanism's _stateRegistry and state type entries
        from PsyNeuLink.Functions.States.State import State_Base
        self._stateRegistry = {}
        # InputState
        from PsyNeuLink.Functions.States.InputState import InputState
        register_category(entry=InputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)
        # ParameterState
        from PsyNeuLink.Functions.States.ParameterState import ParameterState
        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)
        # OutputState
        from PsyNeuLink.Functions.States.OutputState import OutputState
        register_category(entry=OutputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

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
        self._receivesProcessInput = False
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
        # Note: _instantiate_input_states (below) will parse into 1D arrays, one for each input state
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

        self._instantiate_input_states(context=context)

        from PsyNeuLink.Functions.States.ParameterState import instantiate_parameter_states
        instantiate_parameter_states(owner=self, context=context)

        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        # self.instantiate_output_states(context=context)
        from PsyNeuLink.Functions.States.OutputState import instantiate_output_states
        instantiate_output_states(owner=self, context=context)

    def _instantiate_input_states(self, context=None):
        """Call State._instantiate_input_states to instantiate orderedDict of inputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_input_states
        """

        from PsyNeuLink.Functions.States.InputState import _instantiate_input_states
        _instantiate_input_states(owner=self, context=context)

    def _add_projection_to_mechanism(self, state, projection, context=None):

        from PsyNeuLink.Functions.Projections.Projection import add_projection_to
        add_projection_to(receiver=self, state=state, projection_spec=projection, context=context)

    def _add_projection_from_mechanism(self, receiver, state, projection, context=None):
        """Add projection to specified state
        """
        from PsyNeuLink.Functions.Projections.Projection import add_projection_from
        add_projection_from(sender=self, state=state, projection_spec=projection, receiver=receiver, context=context)

    def execute(self, input=None, runtime_params=None, time_scale=TimeScale.TRIAL, context=None):
        """Carry out a single execution of the mechanism.

        Update inputState(s) and param(s), call subclass __execute__, update outputState(s), and assign self.value

        COMMENT:
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
        COMMENT

        Arguments
        ---------

        input : List[value] or ndarray : default variableInstanceDefault
            input to use for execution of the mechanism.
            This must be consistent with the format mechanism's inputState(s):
            the number of items in the outermost level of list,
            or axis 0 of ndarray, must equal the number of inputStates (if there is more than one), and each
            item must be compatible with the format (number and type of elements) of each inputState's variable
            (see :ref:`Run_Inputs` for details of input specification formats).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]:
            dictionary that can include any of the parameters used as arguments to instantiate the object,
            and the function of or projection(s) to any of its states.  Any value assigned to a parameter
            will override the current value of that parameter for this -- but only this execution of the mechanism;
            it will return to its previous value following execution.
            Each entry is either the specification for one of the mechanism's params (in which case the key
            is the name of the param, and its value the value to be assigned to that param), or a dictionary
            for a specified type of state (in which case, the key is the name of a specific state or a keyword
            indicating the type of state (:keyword:`INPUT_STATE_PARAMS`, :keyword:`OUTPUT_STATE_PARAMS` or
            :keyword:`PARAMETER_STATE_PARAMS`), and the value is a dictionary containing parameter dictionaries for that
            state or all states of the specified type).  The latter (state dictionaries) contain
            entries that are themselves dictionaries containing parameters for the state's function or its projections.
            The key for each entry is a keyword indicating whether it is for the state's function
            (:keyword:`FUNCTON_PARAMS`), all of its projections (:keyword:`PROJECTION_PARAMS`), a particular type of
            projection (:keyword:`MAPPING_PARAMS or :keyword:`CONTROL_SIGNAL_PARAMS`), or to a specific projection
            (using its name), and the value of each entry is a dictionary containing the parameters for the function,
            projection, or set of projections (keys of which are parameter names, and values the values to be assigned).

          COMMENT:
            ?? DO PROJECTION DICTIONARIES PERTAIN TO INCOMING OR OUTGOING PROJECTIONS OR BOTH??
            ?? CAN THE KEY FOR A STATE DICTIONARY REFERENCE A SPECIFIC STATE BY NAME, OR ONLY STATE-TYPE??

            state keyword: dict for state's params
                function or projection keyword: dict for funtion or projection's params
                    parameter keyword: vaue of param

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
                        + PROJECTION_PARAMS:<dict>:
                             entry will be passed to all of the State's projections, and used by
                             by their execute methods, overriding their paramInstanceDefaults for that call
                        + MAPPING_PARAMS:<dict>:
                             entry will be passed to all of the State's Mapping projections,
                             along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
                        + CONTROL_SIGNAL_PARAMS:<dict>:
                             entry will be passed to all of the State's ControlSignal projections,
                             along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
                        + <projectionName>:<dict>:
                             entry will be passed to the State's projection with the key's name,
                             along with any in the PROJECTION_PARAMS and Mapping or ControlSignal dicts
          COMMENT

        time_scale : TimeScale :  default TimeScale.TRIAL
            determines whether mechanisms are executed for a single time step or a trial

        Returns
        -------

        output of mechanism : ndarray
            outputState.value containing the output of each of the mechanism's outputStates[]
            after either one time_step or the full trial

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

        Call execute method for each in a sequence of executions specified by the ``inputs`` argument
        (see :ref:`Run_Inputs` in :doc:`Run` for additional details of formatting input specifications)

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_input_value
            input for each execution of mechanism (see :ref:`Run_Inputs` in :doc:`Run` for
            detailed description of formatting requirements and options).

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
        Use mapping of items to outputStates in self._outputStateValueMapping
        Notes:
        * self._outputStateValueMapping must be implemented by Mechanism subclass (typically in its function)
        * if len(self.value) == 1, (i.e., there is only one value), absence of self._outputStateValueMapping is forgiven
        * if the function of a Function is specified only by FUNCTION and returns a value with len > 1
            it MUST also specify kwFunctionOutputStateValueMapping

        """
        if len(self.value) == 1:
            self.outputStates[list(self.outputStates.keys())[0]].value = self.value[0]
        #
        # Assign items in self.value to outputStates using mapping of states to values in self._outputStateValueMapping
        else:
            for state in self.outputStates:
                try:
                    self.outputStates[state].value = self.value[self._outputStateValueMapping[state]]
                except AttributeError:
                    raise MechanismError("{} must implement _outputStateValueMapping attribute in function".
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

    def _get_mechanism_param_values(self):
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

def _is_mechanism_spec(spec):
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
