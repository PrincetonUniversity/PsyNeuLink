# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  InputState *****************************************************
#
"""

Overview
--------

An inputState of a mechanism accepts inputs from projections comming from other mechanisms in a process or system,
and/or the input to the process or system itself (if the mechanism to which the inputState belongs is the
:keyword:`ORIGIN` mechanism [LINK] of that process or system).  It's ``function`` combines the values of these inputs,
and the result is assigned to a corresponding item in the owner mechanism's ``variable``.

.. _InputState_Creation:

Creating an InputState
----------------------

An inputState can be created by calling its constructor, but in general this is not necessary as mechanisms can
usually automatically construct the inputStates they need when they are created.  For example, if the mechanism is
being created within the :ref:`pathway of a process <Process_Pathway>`, its inputState will created and assigned as
the receiver of a MappingProjection from the  preceding mechanism in the pathway. If one or more custom inputStates
need to be specified when a mechanism is created, or added to an existing mechanism, this can be done using the
mechanism's parameter dictionary, in an entry with the key :keyword:`INPUT_STATES` [LINK] and a value that is one or
a list of any of the following:

    * An existing **inputState** object or the name of one.  Its ``value`` must be compatible with item of the owner
      mechanism's ``variable`` to which it will be assigned (see [LINK]).
    ..
    * The :class:`InputState` **class** or a string.  This creates a default inputState using the owner
      mechanism's ``variable`` as the template for the inputState's ``value``. [LINK]  If :keyword:`InputState`
      is used, a default name[LINK] is assigned to the state;  if a string is, it is assigned as the name
      of the inputState.
    ..
    * A **value**.  This creates a default inputState using the specified value as inputState's ``value``.  This must
      be compatible with the owner mechanism's ``variable``.
    ..
    * A **Projection subclass**. This creates a default inputState using the owner mechanism's ``variable`` as
      the template for the inputState's ``value`` [LINK], and a projection of the specified type to the inputState
      also using the owner mechanism's ``variable`` as the template for its ``value``.
    ..
    * A **Projection object**.  This creates a default inputState using the owner mechanism's ``variable`` as
      the template for the inputState's ``value`` [LINK], and assigns the state as the projection's ``receiver``.
      The projection's ``value`` must be compatible with the ``variable`` of the mechanism to which the inputState
      belongs.
    ..
    * A **specification dictionary**.  The inputState is created using the owner mechanism's ``variable`` as
      the template for the inputState's ``value`` [LINK].  In addition to the standard entries of a parameter
      dictionary [LINK], the dictionary can have a :keyword:`STATE_PROJECTIONS` entry, the value of which can be a
      Projection, projection specificadtion dictionary [LINK], or a list containing items thate are either of those.
    ..
    * A :any:`ParamValueProjection`.  This creates a default inputState using the ``value`` item as its ``value``,
      and assigns the state as the ``receiver`` of the ``projection`` item.

    .. note::
       In all cases, the resulting value of the inputState must be compatible (that is, have the same number and type
       of elements) as the item of its owner mechanism's ``variable`` to which it is assigned (see [LINK]).

An inputState must be owned by a mechanism. Therefore, if the inputState is created directly, the mechanism to which it
belongs must be specified in the ``owner`` argument of its constructor; if the inputState is specified in the
:keyword:`INPUT_STATES` entry of the parameter dictionary for a mechanism, then the owner is inferred from the context.

Structure
---------

Every inputState is owned by a :doc:`mechanism <Mechanism>`. It can receive one or more MappingProjections from other
mechanisms, as well as from the process to which its owner belongs (if it is the :keyword:`ORIGIN` [LINK] mechanism
for that process.  A list of projections received by an inputState is kept in its ``receivesFromProjections``
attribute.  Like all PsyNeuLink components, it has the three following fundamental attributes:

* ``variable``:  this serves as a template for the ``value`` of each projection that the inputState receives;
  each must match both the number and types of elements of ``variable``.

* ``function``:  this performs an elementwise (Hadamard) aggregation  of the ``values`` of the projections
   received by the inputState.  The default function is :any:`LinearCombination` that sums the values.
   A custom function can be specified (e.g., to perform a Hadamard product, or to handle non-numeric values in
   some way), so long as it generates an output that is compatible with the ``value`` expected for the inputState
   by the mechanism's ``variable``.  It assigns the result to the inputState's ``value`` attribute.

* ``value``:  this is the aggregated value of the projections received by the inputState, assigned to it by the
  inputState's ``function``.  It must be compatible
  COMMENT:
  both with the inputState's ``variable`` (since the ``function``
  of an inputState only combines the values of its projections, but does not otherwise transform its input),
  COMMENT
  with its corresponding item of the owner mechanism's ``variable``.

Execution
---------

States cannot be executed directly.  They are executed when the mechanism to which they belong is executed. When this
occurs, each inputState executes any projections it receives, calls its ``function`` to aggregate their values, and
then assigns this to its ``value`` attribute.  This is also assigned as the value of the item for the inputState in
the mechanism's ``inputValue`` and ``variable`` attributes (see :ref:`Mechanism InputStates <_Mechanism_Variable>`.

.. _InputState_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state_list
from PsyNeuLink.Components.Functions.Function import *

# InputStatePreferenceSet = ComponentPreferenceSet(log_pref=logPrefTypeDefault,
#                                                          reportOutput_pref=reportOutputPrefTypeDefault,
#                                                          verbose_pref=verbosePrefTypeDefault,
#                                                          param_validation_pref=paramValidationTypeDefault,
#                                                          level=PreferenceLevel.TYPE,
#                                                          name='InputStateClassPreferenceSet')

# class InputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class InputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class InputState(State_Base):
    """Implements subclass of State that calculates and represents the input of a Mechanism

    COMMENT:

        Description
        -----------
            The InputState class is a Component type in the State category of Function,
            Its FUNCTION executes the projections that it receives and updates the InputState's value

        Class attributes
        ----------------
            + componentType (str) = kwInputState
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination, Operation.SUM)
                + FUNCTION_PARAMS (dict)
                # + kwStateProjectionAggregationFunction (LinearCombination, Operation.SUM)
                # + kwStateProjectionAggregationMode (LinearCombination, Operation.SUM)
            + paramNames (dict)

        Class methods
        -------------
            _instantiate_function: insures that function is ARITHMETIC)
            update_state: gets InputStateParams and passes to super (default: LinearCombination with Operation.SUM)

        StateRegistry
        -------------
            All kwInputState are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances


    Arguments
    ---------
    owner : Mechanism
        mechanism to which inputState belongs;  must be specified or determinable from the context in which
        the state is created

    reference_value : number, list or np.ndarray
        component of owner mechanism's ``variable`` attribute that corresponds to the inputState.

    value : number, list or np.ndarray
        used as template for ``variable``*[]:

    function : Function or method : default LinearCombination(operation=SUM),

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the inputState, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a params dict).

    name : str : default InputState-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by StateRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the PreferenceSet for the inputState.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].


    COMMENT

    Attributes
    ----------

    owner : Mechanism
        mechanism to which inputState belongs.

    receivesFromProjections : Optional[List[Projection]]
        a list of the projections received by the inputState.

    variable : number, list or np.ndarray
        the template for the ``value`` of each projection that the inputState receives, each of which must match
        its number and types of elements.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an elementwise (Hadamard) aggregation  of the ``values`` of the projections received by the
        inputState.

    value : number, list or np.ndarray
        the aggregated value of the projections received by the inputState, output of ``function``.

    """

    #region CLASS ATTRIBUTES

    componentType = kwInputState
    paramsType = INPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'InputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION})

    #endregion

    @tc.typecheck
    def __init__(self,
                 owner,
                 reference_value=NotImplemented,
                 value=NotImplemented,
                 function=LinearCombination(operation=SUM),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function, params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of owner (to override assignment of componentName in super.__init__)
        super(InputState, self).__init__(owner,
                                                  value=value,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)

    def _instantiate_function(self, context=None):
        """Insure that function is LinearCombination and that output is compatible with owner.variable

        Insures that function:
            - is LinearCombination (to aggregate projection inputs)
            - generates an output (assigned to self.value) that is compatible with the component of
                owner.function's variable that corresponds to this inputState,
                since the latter will be called with the value of this InputState;

        Notes:
        * Relevant component of owner.function's variable should have been provided
            as reference_value arg in the call to InputState__init__()
        * Insures that self.value has been assigned (by call to super()._validate_function)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """

        super(InputState, self)._instantiate_function(context=context)

        # Insure that function is Function.LinearCombination
        if not isinstance(self.function.__self__, (LinearCombination, Linear)):
            raise StateError("{0} of {1} for {2} is {3}; it must be of LinearCombination or Linear type".
                                      format(FUNCTION,
                                             self.name,
                                             self.owner.name,
                                             self.function.__self__.componentName, ))

        # Insure that self.value is compatible with (relevant item of) self.owner.variable
        if not iscompatible(self.value, self.reference_value):
            raise InputStateError("Value ({0}) of {1} for {2} is not compatible with "
                                           "the variable ({3}) of its function".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.reference_value))
                                                  # self.owner.variable))

def _instantiate_input_states(owner, context=None):
    """Call State._instantiate_state_list() to instantiate orderedDict of inputState(s)

    Create OrderedDict of inputState(s) specified in paramsCurrent[INPUT_STATES]
    If INPUT_STATES is not specified, use self.variable to create a default input state
    When completed:
        - self.inputStates contains an OrderedDict of one or more inputStates
        - self.inputState contains first or only inputState in OrderedDict
        - paramsCurrent[OUTPUT_STATES] contains the same OrderedDict (of one or more inputStates)
        - each inputState corresponds to an item in the variable of the owner's function
        - if there is only one inputState, it is assigned the full value

    Note: State._instantiate_state_list()
              parses self.variable (2D np.array, passed in constraint_value)
              into individual 1D arrays, one for each input state

    (See State._instantiate_state_list() for additional details)

    :param context:
    :return:
    """
    owner.inputStates = _instantiate_state_list(owner=owner,
                                               state_list=owner.paramsCurrent[INPUT_STATES],
                                               state_type=InputState,
                                               state_param_identifier=INPUT_STATES,
                                               constraint_value=owner.variable,
                                               constraint_value_name="function variable",
                                               context=context)

    # Check that number of inputStates and their variables are consistent with owner.variable,
    #    and adjust the latter if not
    for i in range (len(owner.inputStates)):
        input_state = list(owner.inputStates.values())[i]
        try:
            variable_item_is_OK = iscompatible(owner.variable[i], input_state.value)
            if not variable_item_is_OK:
                break
        except IndexError:
            variable_item_is_OK = False
            break

    if not variable_item_is_OK:
        old_variable = owner.variable
        new_variable = []
        for state_name, state in owner.inputStates:
            new_variable.append(state.value)
        owner.variable = np.array(new_variable)
        if owner.verbosePref:
            warnings.warn("Variable for {} ({}) has been adjusted "
                          "to match number and format of its inputStates: ({})".
                          format(old_variable, append_type_to_name(owner),owner.variable))


    # Initialize self.inputValue to correspond to format of owner's variable, and zero it
# FIX: INSURE THAT ELEMENTS CAN BE FLOATS HERE:  GET AND ASSIGN SHAPE RATHER THAN COPY? XXX
    owner.inputValue = owner.variable.copy() * 0.0

    # Assign self.inputState to first inputState in dict
    try:
        owner.inputState = list(owner.inputStates.values())[0]
    except AttributeError:
        owner.inputState = None


