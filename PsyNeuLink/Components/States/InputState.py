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

An InputState receives the input to a Mechanism provided by the Projections to that Mechanism from others in a Process
or System.  If the InputState belongs to an `ORIGIN` Mechanism (see
`role of Mechanisms in Processes and Systems <Mechanism_Role_In_Processes_And_Systems>`), then it receives the input
specified when that Process or System is `run <Run>`.  The Projections received by an InputState are
listed in its `path_afferents <InputState.path_afferents>` attribute. Its
`function <InputState.function>` combines the values of these inputs, and the result is assigned to an item
corresponding to the InputState in the owner Mechanism's :keyword:`variable <Mechanism.Mechanism_Base.variable>` and
`input_value <Mechanism.Mechanism_Base.input_value>` attributes  (see `Mechanism InputStates <Mechanism_InputStates>`
for additional details about the role of input_states in Mechanisms).


.. _InputState_Creation:

Creating an InputState
----------------------

An InputState can be created by calling its constructor, but in general this is not necessary as a Mechanism can
usually automatically create the InputState(s) it needs when it is created.  For example, if the Mechanism is
being created within the :ref:`pathway of a Process <Process_Pathway>`, its InputState will be created and assigned as
the ``receiver`` of a MappingProjection from the  preceding Mechanism in the pathway.

An InputState must be owned by a Mechanism. Therefore, if the InputState is created directly, its Mechanism
must be specified in the ``owner`` argument of its constructor; if the InputState is specified in the
INPUT_STATES entry of the parameter dictionary for a Mechanism, then the owner is inferred from the context.

.. _InputState_Specification

InputState Specification
~~~~~~~~~~~~~~~~~~~~~~~~

If one or more custom input_states need to be specified when a Mechanism is created,
# or added to an existing Mechanism,
they can be specified in an entry of the Mechanism's parameter dictionary, using the key :keyword`INPUT_STATES`
and a value that specifies one or more input_states. For a single InputState, the value can be any of the
specifications in the the list below.  To create multiple input_states, the value of the INPUT_STATES entry
can be either a list, each item of which can be any of the specifications below;  or, it can be an dictionary,
in which the key for each entry is a string specifying the name for the InputState to be created, and its value is
one of the specifications below:

    * An existing **InputState** object or the name of one.  Its ``value`` must be compatible with the item of the
      owner Mechanism's :py:data:`variable <Component.variable>` to which it will be assigned.
    ..
    * The :class:`InputState` **class** or a string.  This creates a default InputState using the the first item of
      the owner Mechanism's ``variable`` as the InputState's :py:data:`variable <InputState.variable>`.
      If INPUT_STATE is used, a default name is assigned to the State;  if a string is, it is assigned as
      the name of the InputState (see :ref:`naming conventions <LINK>`).
    ..
    * A **value**.  This creates a default InputState using the specified value as InputState's ``variable``.
      This must be compatible with the item of the owner Mechanism's ``variable`` to which the InputState is assigned.
    ..
    * A **Projection subclass**. This creates a default InputState using the first item of the owner Mechanism's
      ``variable`` as the InputState's :py:data:`variable <InputState.variable>`, and a Projection of the specified
      type to the InputState using its ``variable`` as the template for the Projection's ``value``.
    ..

       CONFIRM THAT THIS IS TRUE:
    * A **Projection object**.  This creates a default InputState using the first item of the owner Mechanism's
    ``variable`` as the template for the InputState's ``variable``, and assigns the State as the Projection's
    ``receiver``. The Projection's ``value`` must be compatible with the InputState's ``variable``.
    ..
    * A **specification dictionary**.  This creates the specified InputState using the first item of the owner
      Mechanism's ``variable`` as the InputState's :py:data:`variable <InputState.variable>`.  In addition to the
      standard entries of a :ref:`params <LINK>` dictionary, the dictionary can have a PROJECTIONS
      entry, the value of which can be a Projection,
      :ref:`Projection specification dictionary <Projection_In_Context_Specification>`, or a list containing
      items that are either of those.
    ..
    * A **2-item tuple**.  This creates a default InputState using the first (value) item as its
      :keyword:`variable`, and assigns the State as the :keyword:`receiver` of the 2nd (Projection) item.

    .. note::
       In all cases, the resulting ``value`` of the InputState must be compatible with (that is, have the same number
       and type of elements as) the item of its :ref:`owner Mechanism's variable <Mechanism_Variable>` to which it is
       assigned. This is insured by the default ``function`` (:any:`LinearCombination`), since this preserves the
       format of its input;  it must also be true for any other function that is assigned as the ``function`` for an
       InputState.
COMMENT

COMMENT:
   CHECK THIS:
             NUMBER OF STATES MUST EQUAL LENGTH OF MECHANISM'S ATTRIBUTE (VARIABLE OR OUTPUTVALUE)
             SINGLE STATE FOR MULTI-ITEM MECHANISM ATTRIBUTE ASSIGNS (OR AT LEASET CHECKS FOR)
                MULTI-ITEM ATTRIBUTE OF STATE
             MATCH OF FORMATS OF CORRESPONDING ITEMS ARE VALIDATED
             ERROR IS GENERATED FOR NUMBER MISMATCH
             reference_value IS THE ITEM OF variable CORRESPONDING TO THE InputState
COMMENT

COMMENT:
Assigning input_states using the INPUT_STATES entry of a Mechanism's parameter dictionary adds them to any
that are automatically generated for that Mechanism;  if the name of one explicitly specified is the same as one
automatically generated, the name will be suffixed with a numerical index and added (that is, it will *not* replace
the one automatically generated). InputStates can also be added by using the
:py:func:`assign_output_state <OutputState.assign_output_state>`. If the Mechanism requires multiple input_states
(i.e., it's ``variable`` attribute has more than on item), it assigns the ``value`` of each InputState to an item of
its ``variable`` (see :ref:`Mechanism Variable <Mechanism_Variable>`). Therefore, the number of input_states
specified must equal the number of items in the Mechanisms's ``variable``.  An exception is if the Mechanism's
``variable`` has more than one item, it may still be assigned a single InputState;  in that case, the ``value`` of
that InputState must have the same number of items as the  Mechanisms's ``variable``.  For cases in which there are
multiple input_states, the order in which they are specified in the list or OrderedDict must parallel the order of
the items to which they will be assigned in the Mechanism's ``variable``; furthermore, as noted above, the ``value`` for
each InputState must match (in number and types of elements) the item of ``variable`` to which it will be assigned.
COMMENT


.. _InputState_Projections:

Projections
~~~~~~~~~~~

When an InputState is created, it can be assigned one or more `Projections <Projection>`, using either the
**projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument with
the key *PROJECTIONS*.  An InputState can be assigned either `MappingProjection(s) <MappingProjection>` or
`GatingProjection(s) <GatingProjection>`.  MappingProjections are assigned to its `pathway_afferents` attribute
and GatingProjections to its `mod_afferents` attribute.  See `State_Projections` for additional details concerning
the specification of Projections when creating a State.


.. _InputState_Structure:

Structure
---------

Every InputState is owned by a `Mechanism <Mechanism>`. It can receive one or more
`MappingProjections <MappingProjection>` from other Mechanisms, as well as from the Process or System to which its
owner belongs (if it is the `ORIGIN` Mechanism for that Process or System).  The Projections received by an
InputState are listed in its `path_afferents <InputState.path_afferents>` attribute.

Like all PsyNeuLink components, an InputState has the three following core attributes:

* `variable <InputState.variable>`:  this serves as a template for the :keyword:`value` of each Projection that the
  InputState receives: each must match both the number and type of elements of the InputState's
  `variable <InputState.variable>`.
..
* `function <InputState.function>`:  this performs an elementwise (Hadamard) aggregation  of the
  :keyword:`value` of all of the Projections received by the InputState, and assigns the result to the InputState's
  `value <InputState.value>` attribute.  The default function is `LinearCombination` that sums the values.  A custom
  function can be specified (e.g., to perform a Hadamard product, or to handle non-numeric values in some way), so long
  as it generates a result that is compatible with the format of the `value <InputState.value>` of the InputState
  expected by its owner Mechanism's `variable <Mechanism.Mechanism_Base.variable>`.
..
* `value <InputState.value>`:  this is the aggregated value of the Projections received by the InputState, assigned to
  it by the InputState's `function <InputState.function>`.  It must be compatible with item of the owner Mechanism's
  `variable <Mechanism.Mechanism_Base.variable>` to which the InputState has been assigned.

Execution
---------

An InputState cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When this occurs, the InputState executes any Projections it receives, calls its `function <InputState.function>` to
aggregate their values, and then assigns the result to the InputState's `value <InputState.value>` attribute.  This,
in turn, is assigned to the item of the Mechanism's `variable <Mechanism.Mechanism_Base.variable>` and
`input_value <Mechanism.Mechanism_Base.input_value>` attributes corresponding to that InputState
(see `Mechanism variable and input_value attributes <Mechanism_Variable>` for additional details).

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
    function=LinearCombination(operation=SUM), \
    value=None,                                \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Implements a subclass of State that calculates and represents the input to a Mechanism.

    COMMENT:

        Description
        -----------
            The InputState class is a Component type in the State category of Function,
            Its FUNCTION executes the Projections that it receives and updates the InputState's value

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
        the Mechanism to which the InputState belongs;  it must be specified or determinable from the context in which
        the InputState is created.

    reference_value : number, list or np.ndarray
        the value of the item of the owner Mechanism's `variable <Mechanism.Mechanism_Base.variable>` attribute to which
        the InputState is assigned; used as the template for the InputState's `value <InputState.value>` attribute.

    variable : number, list or np.ndarray
        specifies the template for the InputState's `variable <InputState.variable>` attribute.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to aggregate the values of the Projections received by the InputState.
        It must produce a result that has the same format (number and type of elements) as its input.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the InputState or its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default InputState-<index>
        a string used for the name of the InputState.
        If not is specified, a default is assigned by StateRegistry of the Mechanism to which the InputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the InputState.
        If it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism
        the Mechanism to which the InputState belongs.

    afferents : Optional[List[Projection]]
        a list of the Projections received by the InputState
        (i.e., for which it is a `receiver <Projection.Projection.receiver>`).

    variable : number, list or np.ndarray
        the template for the `value <Projection.Projection.value>` of each Projection that the InputState receives,
        each of which must match the format (number and types of elements) of the InputState's :keyword:`variable`.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation of the `value <Projection.Projection.value>` of each
        Projection received by the InputState.

    value : number, list or np.ndarray
        the aggregated value of the Projections received by the InputState; output of `function <InputState.function>`.

    name : str : default <State subclass>-<index>
        the name of the InputState.
        Specified in the **name** argument of the constructor for the OutputState.  If not is specified, a default is
        assigned by the StateRegistry of the Mechanism to which the OutputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the InputState.
        Specified in the **prefs** argument of the constructor for the Projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in ``__init__.py``
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
                 size=None,
                 function=LinearCombination(operation=SUM),
                 weight=None,
                 exponent=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  weight=weight,
                                                  exponent=exponent,
                                                  params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of owner (to override assignment of componentName in super.__init__)
        super(InputState, self).__init__(owner,
                                         variable=variable,
                                         size=size,
                                         projections=projections,
                                         params=params,
                                         name=name,
                                         prefs=prefs,
                                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weights and exponents
        
        This needs to be done here, since paramClassDefault declarations assign None as default
            (so that they can be ignored if not specified here or in the function)
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        
        if WEIGHT in target_set and target_set[WEIGHT] is not None:
            if not isinstance(target_set[WEIGHT], (int, float)):
                raise InputStateError("{} parameter of {} for {} ({}) must be an int or float".
                                      format(WEIGHT, self.name, self.owner.name, target_set[WEIGHT]))

        if EXPONENT in target_set and target_set[EXPONENT] is not None:
            if not isinstance(target_set[EXPONENT], (int, float)):
                raise InputStateError("{} parameter of {} for {} ({}) must be an int or float".
                                      format(EXPONENT, self.name, self.owner.name, target_set[EXPONENT]))


    def _instantiate_function(self, context=None):
        """Insure that function is LinearCombination and that output is compatible with owner.variable

        Insures that function:
            - is LinearCombination (to aggregate Projection inputs)
            - generates an output (assigned to self.value) that is compatible with the component of
                owner.function's variable that corresponds to this InputState,
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

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of State's constructor

        Call _instantiate_projections_to_state to assign:
            PathwayProjections to .pathway_afferents
            ModulatoryProjections to .mod_afferents
        """
        self._instantiate_projections_to_state(projections=projections, context=context)

    def _execute(self, function_params, context):
        """Call self.function with self.variable

        If there were no Transmissive Projections, ignore and return None
        """

        # If there were any Transmissive Projections:
        if self._path_proj_values:
            # Combine Projection values
            combined_values = self.function(variable=self._path_proj_values,
                                            params=function_params,
                                            context=context)
            return combined_values

        # There were no Projections
        else:
            # mark combined_values as none, so that (after being assigned to self.value)
            #    it is ignored in execute method (i.e., not combined with base_value)
            return None

    @property
    def pathway_projections(self):
        return self.path_afferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.path_afferents = assignment


def _instantiate_input_states(owner, context=None):
    """Call State._instantiate_state_list() to instantiate orderedDict of InputState(s)

    Create ContentAddressableList of InputState(s) specified in paramsCurrent[INPUT_STATES]

    If INPUT_STATES is not specified, use self.variable to create a default input state

    When completed:
        - self.input_states contains an OrderedDict of one or more input_states
        - self.input_state contains the `primary InputState <Mechanism_InputStates>`:  first or only one in OrderedDict
        - paramsCurrent[OUTPUT_STATES] contains the same OrderedDict (of one or more input_states)
        - each InputState corresponds to an item in the variable of the owner's function
        - the value of all of the input_states is stored in a list in input_value
        - if there is only one InputState, it is assigned the full value

    Note: State._instantiate_state_list()
              parses self.variable (2D np.array, passed in constraint_value)
              into individual 1D arrays, one for each input state

    (See State._instantiate_state_list() for additional details)
    """

    state_list = _instantiate_state_list(owner=owner,
                                         state_list=owner.input_states,
                                         state_type=InputState,
                                         state_param_identifier=INPUT_STATES,
                                         constraint_value=owner.variable,
                                         constraint_value_name="function variable",
                                         context=context)


    # FIX: 5/23/17:  SHOULD APPEND THIS TO LIST OF EXISTING INPUT_STATES RATHER THAN JUST ASSIGN;
    #                THAT WAY CAN USE INCREMENTALLY IN COMPOSITION
    # if context and 'COMMAND_LINE' in context:
    #     if owner.input_states:
    #         owner.input_states.extend(state_list)
    #     else:
    #         owner.input_states = state_list
    # else:
    #     if owner._input_states:
    #         owner._input_states.extend(state_list)
    #     else:
    #         owner._input_states = state_list

    # FIX: This is a hack to avoid recursive calls to assign_params, in which output_states never gets assigned
    # FIX: Hack to prevent recursion in calls to setter and assign_params
    if context and 'COMMAND_LINE' in context:
        owner.input_states = state_list
    else:
        owner._input_states = state_list



    # Check that number of input_states and their variables are consistent with owner.variable,
    #    and adjust the latter if not
    for i in range (len(owner.input_states)):
        input_state = owner.input_states[i]
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
        for state_name, state in owner.input_states:
            new_variable.append(state.value)
        owner.variable = np.array(new_variable)
        if owner.verbosePref:
            warnings.warn("Variable for {} ({}) has been adjusted "
                          "to match number and format of its input_states: ({})".
                          format(old_variable, append_type_to_name(owner),owner.variable))
