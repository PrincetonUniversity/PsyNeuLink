# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  OutputState *****************************************************

"""

Overview
--------

OutputState(s) represent the result(s) of executing a mechanism.  This may be the result(s) of its ``function`` and/or
other derived values.  The full results are stored in the mechanism's ``value`` attribute;  outputStates
are used to represent individual items of the ``value``, and/or useful quantities derived from them.  For example, the
``function`` of a :doc:`TransferMechanism` generates a result (the transformed value of its input);  however, the
mechanism has outputStates that represent not only that result, but also its mean and
variance (if it is an array).  As a different example, the ``function`` of a :doc:`DDM` mechanism generates several
results (such as decision accuracy and response time), each of which is assigned as the value of a different
outputState.  The outputState(s) of a mechanism can serve as the input to other  mechanisms (by way of
:doc:`projections <Projections>`), or as the output of a process and/or system.  A list of  the  outgoing
projections from an outputState is kept in its ``sendsToProjections`` attribute.

.. _OutputStates_Creation:

Creating an OutputState
-----------------------

An outputState can be created by calling its constructor, but in general this is not necessary as a mechanism
usually creates the outputState(s) it needs automatically when it is created.  For example, if the mechanism is
being created within the :ref:`pathway of a process <Process_Pathway>`, an outputState will be created and assigned
as the ``sender`` of a MappingProjection to the next mechanism in the pathway, or to the process's ``output`` if it is
the :keyword:`TERMINAL` mechanism of that process.

An outputState must be owned by a mechanism. Therefore, if the outputState is created directly,
the mechanism to which it belongs must be specified in the ``owner`` argument of its constructor; if the outputState
is specified in the :keyword:`OUTPUT_STATES` entry of the parameter dictionary for a mechanism, then the owner is
inferred from the context.

If one or more custom outputStates need to be added when a mechanism is created, or added to an existing
mechanism, they can be specified in an entry of the mechanism's parameter dictionary, using the key
:keyword:`OUTPUT_STATES` and a value that specifies the outputState for each one to be added. For a single
outputState, the value can be any of the specifications in the the list below.  To create multiple outputStates,
the value of the  :keyword:`OUTPUT_STATES` entry can be either a list, each item of which can be any of the
specifications below;  or,  it can be an OrderedDict, in which the key for each entry is a string  specifying the
name for the outputState to be  created, and its value is one of the specifications below:

    * An existing **outputState** object.  Its ``variable`` must match (in the number and type of its elements)
      the item of the owner mechanism's ``value`` to which the outputState is assigned
      (designated by its :ref:`index attribute <OutputState_Index>`).
    ..
    * The :class:`OutputState` **class** or a string.  This creates a default outputState that is assigned the first
      item of the owner mechanism's ``value`` as its value.  If :keyword:`OutputState` is used, a default name is
      assigned to the state;  if a string is, it is assigned as the name of the outputState (see [LINK] for naming
      conventions).
    ..
    * A **specification dictionary**.  This can include entries with keys using any of the arguments in an
      outputState's constructor, and a value for that argument.  By default, the outputState is assigned to the
      first item of the owner mechanism's ``value``.  However, the :ref:`index argument <OutputState_Index>`
      can be used to assign the outputState to different item.
    ..
    * A **value**.  This creates a default outputState using the specified value as the outputState's ``variable``.
      This must be compatible with the item of the owner mechanism's ``value`` that will be assigned to the
      outputState (designated by its :ref:`index attribute <OutputState_Index>`).
      COMMENT:
         AT PRESENT THIS IS NOT USEFUL;  HOWEVER, IN THE FUTURE (E.G., WHEN GATING PROJECTIONS TO OUTPUT STATES
         IS ADDED) IT MAY BE USEFUL FOR SPECIFYING A BASEVALUE (E.G., DEFAULT) FOR THE OUTPUTSTATE.
      COMMENT

    .. note::
       In all cases, the ``variable`` of the outputState must be match (that is, have the same number and type
       of elements) as the item of its owner mechanism's ``value`` to which it is assigned (see [LINK]).

.. _OutputState_Index_and_Calculate:

Most mechanisms assign an outputState for each item of their owner mechanism's ``value``, and some assign additional
outputStates that calculate values derived from one more more of those items.  Assigning outputStates
explicitly (i.e., including an :keyword:`OUTPUT_STATES` entry in the mechanism's params dictionary) adds them to any
that are automatically generated for that mechanism;  if the name of one explicitly specified is them same as one
automatically generated, the name will be suffixed with a numerical index and added (that is, it will *not* replace
the one automatically generated). OutputStates can also be added by using the assign_output_state method [LINK]. By
default, a specified outputState  will use the first item of the owner mechanism's ``value``.  However, it can be
assigned a different item by  specifying its ``index`` parameter [LINK]. The ``variable`` of an outputState must match
(in the number and type of  its elements) the item of the mechanism's ``value`` to which it is assigned. An outputState
can also be configured to transform the value of the item, by specifying a function for its ``calculate`` parameter
[LINK];  the function must be able to take as input a value that is compatible with the item to which the
outputState is assigned.

.. _OutputState_Structure:

Structure
---------

Every outputState is owned by a :doc:`mechanism <Mechanism>`. It can send one or more MappingProjections to other
mechanisms;  it can also  be treated as the output of a process or system to which its owner belongs (if it is the
:keyword:`TERMINAL` [LINK] mechanism for that process or system).  A list of projections sent by an outputState is
maintained in its ``sendsToProjections`` attribute.

Like all PsyNeuLink components, it has the three following core attributes:

* ``variable``:  this must match (both in number and types of elements) the value of the item of its owner mechanism's
  ``value`` attribute to which it is assigned (in its ``index`` attribute).

* ``function``: this is implemented for potential future use, but is not actively used by PsyNeuLink at the moment.

* ``value``:  this is assigned the result of the outputState`s ``function``, possibly modifed by its ``calculate``
  parameter, and used as the input to any projections that it sends.

An outputState also has two additional attributes that determine its operation:

.. _OutputState_Index:

* ``index``: this determines the item of its owner mechanism's ``value`` to which it is assigned.  By default, this is
  set to 0, which assigns it to the first item.

.. _OutputState_Calculate:

* ``calculate``:  this specifies the function used to convert the item to which the outputState is assigned to
  the outputState's value.  The result is assigned to the outputState's ``value`` attribute. The default for
  ``calculate`` is the identity function, which simply assigns the item of the mechanism'sm ``value`` unmodified as
  the ``value`` of the outputState.  However, it can be assigned any function that can take as input the  value of
  the item to which the outputState is assigned.  Note that the ``calculate`` function is distinct from the
  outputState's ``function`` parameter (which is reserved for future use).

.. _OutputState_Execution:

Execution
---------

An outputState cannot be executed directly.  It is executed when the mechanism to which it belongs is executed.
When this occurs, the mechanism places the results of its execution in its ``value`` attribute, and the value of the
outputState is then updated by calling its ``calculate`` function using as its input the item of the onwer mechanism's
``value`` to which the outputState is assigned.  The result is assigned to the outputState's ``value``, as well as
to a corresponding item of the mechanism's ``outputValue`` attribute.  It is also used as the input to any projections
for which the outputState is the sender.

.. _OutputState_Class_Reference:

Class Reference
---------------


"""

# import Components
from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state_list
from PsyNeuLink.Components.Functions.Function import *

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


PRIMARY_OUTPUT_STATE = 0

class OutputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OutputState(State_Base):
    """
    OutputState(                               \
    owner,                                     \
    value=None,                                \
    index=PRIMARY_OUTPUT_STATE,                \
    calculate=Linear,                          \
    function=LinearCombination(operation=SUM), \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Implements subclass of State that represents the output of a mechanism

    COMMENT:

        Description
        -----------
            The OutputState class is a type in the State category of Component,
            It is used primarily as the sender for MappingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = OUTPUT_STATES
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS   (Operation.PRODUCT)
            + paramNames (dict)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: LinearCombination with Operation.SUM)

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the mechanism to which the outputState belongs; it must be specified or determinable from the context in which
        the outputState is created.

    reference_value : number, list or np.ndarray
        a template for the item of the owner mechanism's ``value`` attribute to which the outputState will be assigned
        (specified by the ``index`` argument).  This must match (in number and type of elements) the ``variable``
        argument; it is used to insure the compatibility of the source of the input to the outputState and its
        ``variable`` (used for its ``function`` and ``calculate`` routines).

    value : number, list or np.ndarray
        used as the template for ``variable``.

    index : int : default PRIMARY_OUTPUT_STATE
        the item in the owner mechanism's ``value`` attribute used as input of the ``calculate`` function, to determine
        the ``value`` of the outputState.

    calculate : function or method : default Linear
        used to convert item of owner mechanism's ``value`` to outputState's ``value`` (and corresponding
        item of owner's ``outputValue``.  It must accept a value that has the same format as the mechanism's ``value``.

    function : Function or method : default LinearCombination(operation=SUM)
        function used to aggregate the values of the projections received by the outputState.
        It must produce a result that has the same format (number and type of elements) as its ``value``.
        It is implemented for consistency with other states, but is not actively used by PsyNeuLInk at the moment
        (see note under a description of the ``function`` attribute below).

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the outputState, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Component` for specification of a params dict).

    name : str : default OutputState-<index>
        a string used for the name of the outputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the outputState
        belongs (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the PreferenceSet for the outputState.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].


    Attributes
    ----------

    owner : Mechanism
        the mechanism to which the outputState belongs.

    sendsToProjections : Optional[List[Projection]]
        a list of the projections sent by the outputState (i.e., for which the outputState is a ``sender``).

    variable : number, list or np.ndarray
        assigned an item of the ``outputValue`` of its owner mechanism.

    index : int
        the item in the owner mechanism's ``value`` attribute used as input of the ``calculate`` function, to determine
        the ``value`` of the outputState.

    calculate : function or method : default Linear
        function used to convert the item of owner mechanism's ``value`` specified by the ``index`` attribute;  it is
        combined with the result of the outputState's ``function`` to determine it's ``value``, and the corresponding
        item of the owner mechanism's ``outputValue``. Default is Linear (identity function) which simply transfers the
        value as is.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation  of the ``values`` of the projections received by the
        outputState.  The result is combined with the result of the calculate function and assigned as the ``value``
        of the outputState, and the corresponding item of the owner's ``outputValue``.

        .. note::
           Currently PsyNeuLink does not support projections to outputStates.  The ``function`` attribute is
           implemented for consistency with other states classes, and for potential future use.  The default simply
           passes its input to its output. The ``function`` attribute can be modified to change this behavior.
           However, for compatibility with future versions, it is *strongly* recommended that such functionality
           be implemented by assigning the desired function to the ``calculate`` attribute; this will insure
           compatibility with future versions.

    value : number, list or np.ndarray
        assigned the result of the ``calculate`` function, combined with any result of the outputState's ``function``,
        which is also assigned to the corresopnding item of the owner mechanism's ``outputValue``.

    name : str : default <State subclass>-<index>
        name of the outputState.
        Specified in the name argument of the call to create the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the PreferenceSet for the outputState.
        Specified in the prefs argument of the call to create the projection;  if it is not specified, a default is
        assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATES
    paramsType = OUTPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION})
    #endregion

    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value,
                 variable=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate:function_type=Linear,
                 function=LinearCombination(operation=SUM),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(index=index,
                                                  calculate=calculate,
                                                  function=function,
                                                  params=params)

        self.reference_value = reference_value

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def _validate_variable(self, variable, context=None):
        """Insure variable is compatible with output component of owner.function relevant to this state

        Validate self.variable against component of owner's value (output of owner's function)
             that corresponds to this outputState (since that is what is used as the input to OutputState);
             this should have been provided as reference_value in the call to OutputState__init__()

        Note:
        * This method is called only if the parameterValidationPref is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """
        super(OutputState,self)._validate_variable(variable, context)

        self.variableClassDefault = self.reference_value

        # Insure that self.variable is compatible with (relevant item of) output value of owner's function
        if not iscompatible(self.variable, self.reference_value):
            raise OutputStateError("Value ({0}) of outputState for {1} is not compatible with "
                                           "the output ({2}) of its function".
                                           format(self.value,
                                                  self.owner.name,
                                                  self.reference_value))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate index and anaylze parameters

        Validate that index is within the range of the number of items in the owner mechanism's ``value``,
        and that the corresponding item is a valid input to the calculate function


        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        try:
            self.owner.value[target_set[INDEX]]
        except IndexError:
            raise OutputStateError("Value of {} argument for {} is greater than the number of items in "
                                   "the outputValue ({}) for its owner mechanism ({})".
                                   format(INDEX, self.name, self.owner.outputValue, self.owner.name))

        # IMPLEMENT: VALIDATE THAT CALCULATE FUNCTION ACCEPTS VALUE CONSISTENT WITH
        #            CORRESPONDING ITEM OF OWNER MECHANISM'S VALUE
        try:
            if isinstance(target_set[CALCULATE], type):
                function = target_set[CALCULATE]().function
            else:
                function = target_set[CALCULATE]
            try:
                function(self.owner.value[target_set[INDEX]])
            except:
                raise OutputStateError("Item {} of value for {} ({}) is not compatible with the function specified for "
                                       "the {} parameter of {} ({})"
                                       "".format(target_set[INDEX],
                                                 self.owner.name,
                                                 self.owner.value[target_set[INDEX]],
                                                 CALCULATE,
                                                 self.name,
                                                 target_set[CALCULATE]))
        except KeyError:
            pass

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate calculate function
        """
        super()._instantiate_attributes_after_function(context=context)

        if isinstance(self.calculate, type):
            self.calculate = self.calculate().function


    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):

        super().update(params=params, time_scale=time_scale, context=context)

        # IMPLEMENT: INCORPORATE paramModulationOperation HERE, AS PER PARAMETER STATE

        if not self.value:
            # # MODIFIED 12/8/16 OLD:
            # self.value = self.calculate(self.owner.value[self.index])
            # MODIFIED 12/8/16 NEW:
            self.value = type_match(self.calculate(self.owner.value[self.index]), type(self.owner.value[self.index]))
            # MODIFIED 12/8/16 END


def _instantiate_output_states(owner, context=None):
    # MODIFIED 12/7/16 NEW:
    # ADD TO DOCUMENTATION BELOW:
    # EXPAND constraint_value to match specification of outputStates (by # and function return values):
    #            in order to both constrain spec and also match # states to # items in constraint
    #            (checked in _instantiate_state_list)
    # For each outputState:
    #      check for index param:
    #          if it is a state, get from attribute
    #          if it is dict, look for param
    #          if it is anything else, assume index is PRIMARY_OUTPUT_STATE
    #      get indexed value from output.value
    #      append the indexed value to constraint_value

    # ALSO: INSTANTIATE CALCULATE FUNCTION
    # MODIFIED 12/7/16 END
    """Call State._instantiate_state_list() to instantiate orderedDict of outputState(s)

    Create OrderedDict of outputState(s) specified in paramsCurrent[INPUT_STATES]
    If INPUT_STATES is not specified, use self.variable to create a default output state
    When completed:
        - self.outputStates contains an OrderedDict of one or more outputStates
        - self.outputState contains first or only outputState in OrderedDict
        - paramsCurrent[OUTPUT_STATES] contains the same OrderedDict (of one or more outputStates)
        - each outputState corresponds to an item in the output of the owner's function
        - if there is only one outputState, it is assigned the full value

    (See State._instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[OUTPUT_STATES] (self.value) is assigned here
        rather than in _validate_params, as it requires function to have been instantiated first

    :param context:
    :return:
    """

    constraint_value = []
    owner_value = np.atleast_2d(owner.value)

    if owner.paramsCurrent[OUTPUT_STATES]:
        for output_state in owner.paramsCurrent[OUTPUT_STATES]:
            # Default is PRIMARY_OUTPUT_STATE
            index = PRIMARY_OUTPUT_STATE
            # If output_state is already an OutputState object, get its index attribute
            if isinstance(output_state, OutputState):
                index = output_state.index
            # If output_state is a specification dict, get its INDEX attribute if specified
            elif isinstance(output_state, dict):
                try:
                    index = output_state[INDEX]
                except KeyError:
                    pass
            constraint_value.append(owner_value[index])
    else:
        constraint_value = owner_value

    owner.outputStates = _instantiate_state_list(owner=owner,
                                                state_list=owner.paramsCurrent[OUTPUT_STATES],
                                                state_type=OutputState,
                                                state_param_identifier=OUTPUT_STATES,
                                                constraint_value=constraint_value,
                                                constraint_value_name="output",
                                                context=context)
    # Assign self.outputState to first outputState in dict
    owner.outputState = list(owner.outputStates.values())[0]

