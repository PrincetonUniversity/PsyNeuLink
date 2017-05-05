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

An inputState receives the input to a mechanism provided by the projections to that mechanism from others in a process
or system.  If the inputState belongs to an `ORIGIN` mechanism (see
`role of mechanisms in processes and systems <Mechanism_Role_In_Processes_And_Systems>`), then it receives the input
specified when that process or system is `run <Run>`.  The projections received by an inputState are
listed in its `receivesFromProjections <InputState.receivesFromProjections>` attribute. Its
`function <InputState.function>` combines the values of these inputs, and the result is assigned to an item
corresponding to the inputState in the owner mechanism's :keyword:`variable <Mechanism.Mechanism_Base.variable>` and
`inputValue <Mechanism.Mechanism_Base.inputValue>` attributes  (see `Mechanism InputStates <Mechanism_InputStates>`
for additional details about the role of inputStates in mechanisms).


.. _InputState_Creation:

Creating an InputState
----------------------

InputStates are created automatically when a mechanism is created.  For example, if a mechanism is created within
the `pathway` of a process`, the inputState for that mechanism will be created and assigned as the
`receiver <MappingProjection.MappingProjection.receiver>` of a `MappingProjection` from the  preceding mechanism in the
pathway;  and a `ControlMechanism <ControlMechanism>` automatically creates an inputState for each mechanism that it
monitors. PsyNeuLink does not currently support the explicit creation of inputStates (this may be implemented in the
future). However they can modified as described below.

COMMENT:
An inputState can be created by calling its constructor, but in general this is not necessary as a mechanism can
usually automatically create the inputState(s) it needs when it is created.  For example, if the mechanism is
being created within the :ref:`pathway of a process <Process_Pathway>`, its inputState will be created and assigned as
the ``receiver`` of a MappingProjection from the  preceding mechanism in the pathway.

An inputState must be owned by a mechanism. Therefore, if the inputState is created directly, its mechanism
must be specified in the ``owner`` argument of its constructor; if the inputState is specified in the
INPUT_STATES entry of the parameter dictionary for a mechanism, then the owner is inferred from the context.

If one or more custom inputStates need to be specified when a mechanism is created, or added to an existing mechanism,
they can be specified in an entry of the mechanism's parameter dictionary, using the key :keyword`INPUT_STATES`
and a value that specifies one or more inputStates. For a single inputState, the value can be any of the
specifications in the the list below.  To create multiple inputStates, the value of the INPUT_STATES entry
can be either a list, each item of which can be any of the specifications below;  or, it can be an OrderedDict,
in which the key for each entry is a string specifying the name for the inputState to be created, and its value is
one of the specifications below:

    * An existing **inputState** object or the name of one.  Its ``value`` must be compatible with the item of the
      owner mechanism's :py:data:`variable <Component.variable>` to which it will be assigned.
    ..
    * The :class:`InputState` **class** or a string.  This creates a default inputState using the the first item of
      the owner mechanism's ``variable`` as the inputState's :py:data:`variable <InputState.variable>`.
      If INPUT_STATE is used, a default name is assigned to the state;  if a string is, it is assigned as
      the name of the inputState (see :ref:`naming conventions <LINK>`).
    ..
    * A **value**.  This creates a default inputState using the specified value as inputState's ``variable``.
      This must be compatible with the item of the owner mechanism's ``variable`` to which the inputState is assigned.
    ..
    * A **Projection subclass**. This creates a default inputState using the first item of the owner mechanism's
      ``variable`` as the inputState's :py:data:`variable <InputState.variable>`, and a projection of the specified
      type to the inputState using its ``variable`` as the template for the projection's ``value``.
    ..

       CONFIRM THAT THIS IS TRUE:
    * A **Projection object**.  This creates a default inputState using the first item of the owner mechanism's
    ``variable`` as the template for the inputState's ``variable``, and assigns the state as the projection's
    ``receiver``. The projection's ``value`` must be compatible with the inputState's ``variable``.
    ..
    * A **specification dictionary**.  This creates the specified inputState using the first item of the owner
      mechanism's ``variable`` as the inputState's :py:data:`variable <InputState.variable>`.  In addition to the
      standard entries of a :ref:`params <LINK>` dictionary, the dictionary can have a STATE_PROJECTIONS
      entry, the value of which can be a Projection,
      :ref:`projection specification dictionary <Projection_In_Context_Specification>`, or a list containing
      items that are either of those.
    ..
    * A :any:`ParamValueProjection` tuple.  This creates a default inputState using the ``value`` item as its
    ``variable``, and assigns the state as the ``receiver`` of the projection item.

    .. note::
       In all cases, the resulting ``value`` of the inputState must be compatible with (that is, have the same number
       and type of elements as) the item of its :ref:`owner mechanism's variable <Mechanism_Variable>` to which it is
       assigned. This is insured by the default ``function`` (:any:`LinearCombination`), since this preserves the
       format of its input;  it must also be true for any other function that is assigned as the ``function`` for an
       inputState.
COMMENT

COMMENT:
   CHECK THIS:
             NUMBER OF STATES MUST EQUAL LENGTH OF MECHANISM'S ATTRIBUTE (VARIABLE OR OUTPUTVALUE)
             SINGLE STATE FOR MULTI-ITEM MECHANISM ATTRIBUTE ASSIGNS (OR AT LEASET CHECKS FOR)
                MULTI-ITEM ATTRIBUTE OF STATE
             MATCH OF FORMATS OF CORRESPONDING ITEMS ARE VALIDATED
             ERROR IS GENERATED FOR NUMBER MISMATCH
             reference_value IS THE ITEM OF variable CORRESPONDING TO THE inputState
COMMENT

COMMENT:
Assigning inputStates using the INPUT_STATES entry of a mechanism's parameter dictionary adds them to any
that are automatically generated for that mechanism;  if the name of one explicitly specified is them same as one
automatically generated, the name will be suffixed with a numerical index and added (that is, it will *not* replace
the one automatically generated). InputStates can also be added by using the
:py:func:`assign_output_state <OutputState.assign_output_state>`. If the mechanism requires multiple inputStates
(i.e., it's ``variable`` attribute has more than on item), it assigns the ``value`` of each inputState to an item of
its ``variable`` (see :ref:`Mechanism Variable <Mechanism_Variable>`). Therefore, the number of inputStates
specified must equal the number of items in the mechanisms's ``variable``.  An exception is if the mechanism's
``variable`` has more than one item, it may still be assigned a single inputState;  in that case, the ``value`` of
that inputState must have the same number of items as the  mechanisms's ``variable``.  For cases in which there are
multiple inputStates, the order in which they are specified in the list or OrderedDict must parallel the order of
the items to which they will be assined in the mechanism's ``variable``; furthemore, as noted above, the ``value`` for
each inputState must match (in number and types of elements) the item of ``variable`` to which it will be assigned.
COMMENT

.. _InputState_Structure:

Structure
---------

Every inputState is owned by a `mechanism <Mechanism>`. It can receive one or more
`MappingProjections <MappingProjection>` from other mechanisms, as well as from the process or system to which its
owner belongs (if it is the `ORIGIN` mechanism for that process or system).  The projections received by an
inputState are listed in its `receivesFromProjections <InputState.receivesFromProjections>` attribute.

Like all PsyNeuLink components, an inputState has the three following core attributes:

* `variable <InputState.variable>`:  this serves as a template for the :keyword:`value` of each projection that the
  inputState receives: each must match both the number and type of elements of the inputState's
  `variable <InputState.variable>`.
..
* `function <InputState.function>`:  this performs an elementwise (Hadamard) aggregation  of the
  :keyword:`value` of all of the projections received by the inputState, and assigns the result to the inputState's
  `value <InputState.value>` attribute.  The default function is `LinearCombination` that sums the values.  A custom
  function can be specified (e.g., to perform a Hadamard product, or to handle non-numeric values in some way), so long
  as it generates a result that is compatible with the format of the `value <InputState.value>` of the inputState
  expected by its owner mechanism's `variable <Mechanism.Mechanism_Base.variable>`.
..
* `value <InputState.value>`:  this is the aggregated value of the projections received by the inputState, assigned to
  it by the inputState's `function <InputState.function>`.  It must be compatible with item of the owner mechanism's
  `variable <Mechanism.Mechanism_Base.variable>` to which the inputState has been assigned.

Execution
---------

An inputState cannot be executed directly.  It is executed when the mechanism to which it belongs is executed.
When this occurs, the inputState executes any projections it receives, calls its `function <InputState.function>` to
aggregate their values, and then assigns the result to the inputState's `value <InputState.value>` attribute.  This,
in turn, is assigned to the item of the mechanism's `variable <Mechanism.Mechanism_Base.variable>` and
`inputValue <Mechanism.Mechanism_Base.inputValue>` attributes corresponding to that inputState
(see `mechanism variable and inputValue attributes <Mechanism_Variable>` for additional details).

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
    """
    InputState(                                \
    owner,                                     \
    reference_value=None,                      \
    value=None,                                \
    function=LinearCombination(operation=SUM), \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Implements a subclass of State that calculates and represents the input to a mechanism.

    COMMENT:

        Description
        -----------
            The InputState class is a Component type in the State category of Function,
            Its FUNCTION executes the projections that it receives and updates the InputState's value

        Class attributes
        ----------------
            + componentType (str) = INPUT_STATE
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
            All INPUT_STATE are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the mechanism to which the inputState belongs;  it must be specified or determinable from the context in which
        the inputState is created.

    reference_value : number, list or np.ndarray
        the value of the item of the owner mechanism's `variable <Mechanism.Mechanism_Base.variable>` attribute to which
        the inputState is assigned; used as the template for the inputState's `value <InputState.value>` attribute.

    value : number, list or np.ndarray
        specifies the template for the inputState's `variable <InputState.variable>` attribute (since an inputState's
        `variable <InputState.variable>` and `value <InputState.value>` attributes must have the same format
        (number and type of elements).

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to aggregate the values of the projections received by the inputState.
        It must produce a result that has the same format (number and type of elements) as its input.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the inputState or its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default InputState-<index>
        a string used for the name of the inputState.
        If not is specified, a default is assigned by StateRegistry of the mechanism to which the inputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the inputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism
        the mechanism to which the inputState belongs.

    receivesFromProjections : Optional[List[Projection]]
        a list of the projections received by the inputState
        (i.e., for which it is a `receiver <Projection.Projection.receiver>`).

    variable : number, list or np.ndarray
        the template for the `value <Projection.Projection.value>` of each projection that the inputState receives,
        each of which must match the format (number and types of elements) of the inputState's :keyword:`variable`.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation of the `value <Projection.Projection.value>` of each
        projection received by the inputState.

    value : number, list or np.ndarray
        the aggregated value of the projections received by the inputState; output of `function <InputState.function>`.

    name : str : default <State subclass>-<index>
        the name of the inputState.
        Specified in the **name** argument of the constructor for the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the inputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = INPUT_STATE
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
                 reference_value=None,
                 variable=None,
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
                                                  variable=variable,
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
        * Relevant item of owner.function's variable should have been provided
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
        - self.inputState contains the `primary inputState <Mechanism_InputStates>`:  first or only one in OrderedDict
        - paramsCurrent[OUTPUT_STATES] contains the same OrderedDict (of one or more inputStates)
        - each inputState corresponds to an item in the variable of the owner's function
        - the value of all of the inputStates is stored in a list in inputValue
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
# FIX:  IS THIS A LIST OR np.array (SHOULD BE A LIST)
    # ??REPLACE THIS WITH owner.inputValue = list(owner.variable) * 0.0??
    owner.inputValue = owner.variable.copy() * 0.0

    # Assign self.inputState to first inputState in dict
    try:
        owner.inputState = list(owner.inputStates.values())[0]
    except AttributeError:
        owner.inputState = None


