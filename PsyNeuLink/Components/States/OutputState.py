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

OutputState(s) represent the result(s) of executing a mechanism.  This may be the result(s) of its
`function <OutputState.function>` and/or values derived from that result.  The full set of results are stored in the
mechanism's `value <Mechanism.Mechanism_Base.value>` attribute.  OutputStates are used to represent individual items
of the mechanism's `value <Mechanism.Mechanism_Base.value>`, and/or useful quantities derived from them.  For example,
the `function <TransferMechanism.TransferMechanism.function>` of a `TransferMechanism` generates a single result (the
transformed value of its input);  however, a TransferMechanism has outputStates that represent not only that
result, but also its mean and variance (if it is an array).  In contrast, the `function <DDM.DDM.function>` of a
`DDM` mechanism generates several results (such as decision accuracy and response time), each of which is assigned as
the `value <OutputState.value>` of a different outputState.  The outputState(s) of a mechanism can serve as the input to
other  mechanisms (by way of `projections <Projections>`), or as the output of a process and/or system.  The
outputState's `sendsToProjections <OutputState.sendsToProjections>` attribute lists all of its outgoing projections.

.. _OutputStates_Creation:

Creating an OutputState
-----------------------

An outputState can be created by calling its constructor. However, in general this is not necessary as a mechanism
usually automatically creates the outputState(s) it needs when it is created.  For example, if the mechanism is
created within the `pathway` of a `process <Process>`, an outputState will be created and assigned as the
`sender <MappingProjection.MappingProjection.sender>` of a `MappingProjection` to the next mechanism in the pathway,
or to the process's `output <Process_Input_And_Ouput>` if the mechanism is a `TERMINAL` mechanism for that process.

An outputState must be owned by a mechanism. Therefore, if the outputState is created explicitly, the mechanism to
which it belongs must be specified in the **owner** argument of its constructor; if the outputState is specified
in the OUTPUT_STATES entry of the `parameter dictionary <ParameterState_Specifying_Parameters>` for a
mechanism, then the owner is inferred from the context.

.. _OutputState_Primary:

Every mechanism has at least one outputState, referred to as its *primary outputState*, that is automatically created
and assigned to the mechanism's `outputState <Mechanism.Mechanism_Base.outputState>` attribute (note the singular),
and also as the first entry in the OrderedDictionary of the mechanism's
`outputStates <Mechanism.Mechanism_Base.outputStates>` attribute (note the plural).  The `value <OutputState.value>` of
the primary outputState is assigned as the first (and often only) item of the mechanism's
`value <Mechanism.Mechanism_Base.value>`, which is the result of the mechanism`s
`function <Mechanism.Mechanism_Base.function>`.  In addition to the primary outputState, many mechanisms also assign
an outputState for each additional item of their `value <Mechanism.Mechanism_Base.value>`, and some assign
additional outputStates that calculate values derived from one or more of those items.

.. _OutputState_Specification:

If one or more custom outputStates need to be added when a mechanism is created, or added to an existing
mechanism, they can be specified in an entry of the mechanism's
`parameter dictionary <ParameterState_Specifying_Parameters>`, using the key OUTPUT_STATES.  For a single
outputState, the value of the entry can be any of the specifications in the the list below.  To create multiple
outputStates, the value can be either: a list, each item of which can be any of the specifications below; or
it can be an OrderedDict in which the key for each entry is a string  specifying the name for the outputState to be
created, and its value is one of the specifications below.  Each outputState to be created can be specified using any
of the following formats:

    * A reference to an **existing outputState**.  Its `variable <OutputState.variable>` must match (in the
      number and type of its elements) the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` to
      which the outputState is assigned (designated by its `index <OutputState_Index>` attribute).
    ..
    * A reference to the **OutputState class** or a ** name string**.  This creates a default outputState that is
      assigned the first item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` as its value.  If the
      name of the OutputState class or its keyword (OUTPUTSTATE) are used, a default name is assigned to the
      outputState;  if a string is used, it is assigned as the name of the outputState
      (see :ref:`naming conventions <LINK>`).
    ..
    * A **specification dictionary**.  This can include entries, each of which uses the name of an argument for
      an outputState parameter (used in the outputState constructor) as its key, and the value of that
      parameter as its value.  By default, the outputState is assigned to the first item of the owner mechanism's
      `value <Mechanism.Mechanism_Base.value>`.  However, :keyword:`INDEX <OutputState_Index>` can be used to
      specify the outputState's  `index <OutputState_Index>` attribute and assign it to different item of the
      mechanism's `value <Mechanism.Mechanism_Base.value>`.
    ..
    * A **value**.  This creates a default outputState using the specified value as the outputState's
      `variable <OutputState.value>`.  This must be compatible with (have the same number and type of elements as) the
      item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` to which the outputState will be assigned
      (its primary outputState by default, or designated by its `index <OutputState.index>` attribute).
      COMMENT:
         AT PRESENT THIS IS NOT USEFUL;  HOWEVER, IN THE FUTURE (E.G., WHEN GATING PROJECTIONS TO OUTPUT STATES
         IS ADDED) IT MAY BE USEFUL FOR SPECIFYING A BASEVALUE (E.G., DEFAULT) FOR THE OUTPUTSTATE.
      COMMENT

    .. note::
       In all cases, the `variable <OutputState.variable>` of the outputState must match (have the same number and
       type of elements) as the item of its owner mechanism's `value <Mechanism.Mechanism_Base.value>` to which it is
       assigned.

COMMENT:
    OutputStates can also be added by using the :py:func:`assign_output_state <OutputState.assign_output_state>` method.
COMMENT

.. _OutputState_Names:

Assigning outputStates explicitly (i.e., including an OUTPUT_STATES entry in the mechanism's
`parameter dictionary <ParameterState_Specifying_Parameters>`) adds them to any that are automatically generated for
that mechanism.  If the name of an explicitly specified outputState is the same as one that was created automatically
(or another one that was created explicitly), its name will be suffixed with a numerical index (incremented for each
outputState with that name), and the outputState will be added to the list (that is, it will *not* replace ones that
were already created).

.. _OutputState_Index_and_Calculate:

By default, an explicitly specified outputState will use the first item of the owner mechanism's
`value <Mechanism.Mechanism_Base.value>`.  However, it can be assigned to a different item by specifying its
`index <OutputState.index>` attribute.  The `variable <OutputState.variable>` of
an outputState must match (in the number and type of its elements) the item of the mechanism's
`value <Mechanism.Mechanism_Base.value>` to which it is assigned. An outputState can also be configured to transform
the value of the item, by specifying a function for its `calculate <OutputState.calculate>` attribute;  the function
must be able to take as input a value that is compatible with the item of the Mechahnism's
`value <Mechanism.Mechanism_Base.value>` to which the outputState is assigned.

.. _OutputState_Structure:

Structure
---------

Every outputState is owned by a `mechanism <Mechanism>`. It can send one or more
`MappingProjections <MappingProjection>` to other mechanisms.  If its owner is a `TERMINAL` mechanism of a process
and/or system, then the outputState will also be treated as the output of that `process <Process_Input_And_Ouput>`
and/or of a system.  The projections that the outputState sends are listed in its
`sendsToProjections <OutputState.sendsToProjections>` attribute.

Like all PsyNeuLink components, an outputState has the three following core attributes:

* `variable <OutputState.variable>`:  this must match (both in number and types of elements) the
  value of the item of its owner mechanism's `value <Mechanism.Mechanism_Base.value>` to which it is assigned
  (designated by its `index <OutputState.index>` attribute).
..
* `function <OutputState.function>`: this aggregates the values of any projections received by the outputState,
  which are combined with the result of the function specified by the outputState's `calculate <OutputState_Calculate>`
  attribute, and then assigned as the outputState's `value <OutputState.value>`.

  .. OutputState_Function_Note_1:
  .. note::
       At present the `function <OutputState.function>` of an outputState is not used, and the outputState's
       `value <OutputState.value>` is determined exclusively by the function specified for its `calculate
       <OutputState_Calculate>` attribute (see `note <OutputState_Function_Note_2>` for details).
  COMMENT:
     SEE update() METHOD FOR NOTES ON FUTURE IMPLEMENTATION OF FUNCTION.
  COMMENT
..
* `value <OutputState.value>`:  this is assigned the result of the function specified by the
  `calculate <OutputState.calculate>` attribute, possibly modified by the result of the outputState`s
  `function <OutputState.function>` (though see `note <OutputState_Function_Note_2>`).  It is used as the input to any
  projections that the outputStatue sends.

.. _OutputState_Attributes:

An outputState also has two additional attributes that determine its operation:

.. _OutputState_Index:

* `index <OutputState.index>`: this determines the item of its owner mechanism's
  `value <Mechanism.Mechanism_Base.value>` to which it is assigned.  By default, this is set to 0, which assigns it to
  the first item of the mechanism's `value <Mechanism.Mechanism_Base.value>`.

.. _OutputState_Calculate:

* `calculate <OutputState.calculate>`:  this specifies the function used to convert the item of the owner mechanism's
  `value <Mechanism.Mechanism_Base.value>` (designated by the outputState's `index <OutputState.index>` attribute),
  before assigning it as the outputState's `value <OutputState.vaue>`.  The result is combined with the result of the
  outputState's `function <OutputState.function>` attribute (which aggregates the value of its projections), to
  determine the outputState's `value <OutputState.value>` (though see `note <OutputState_Function_Note_1>`). The
  default for `calculate  <OutputState.calculate>` is an identity function (`Linear` with **slope**\ =1 and
  **intercept**\ =0), which simply assigns the specified item of the mechanism's
  `value <Mechanism.Mechanism_Base.value>` unmodified as the `value <OutputState.value>` of the outputState. However,
  `calculate  <OutputState.calculate>` can be assigned any function that can take as its input the designated item
  of the owner mechanism's `value <Mechanism.Mechanism_Base.value>`, and the result of which can be combined with the
  result of the outputState's `function <OutputState.function>`.

.. _OutputState_Execution:

Execution
---------

An outputState cannot be executed directly.  It is executed when the mechanism to which it belongs is executed.
When the mechanism is executed, it places the results of its execution in its `value <Mechanism.Mechanism_Base.value>`
attribute. The outputState's `index <OutputState.index>` attribute designates one item of the mechanmism's
`value <Mechanism.Mechanism_Base.value>` for use by the outputState.  The outputState is updated by calling the function
specified by its `calculate <OutputState_Calculate>` attribute with the designated item of the mechanism's
`value <Mechanism.Mechanism_Base.value>` as its input.  This is possibly modified by the result of the outputState's
`function <OutputState.function>` (though see `note <OutputState_Function_Note_2>`).  The final result is assigned as
the outputState's `value <OutputState.value>`, as well as to a corresponding item of the mechanism's
`output_values  <Mechanism.Mechanism_Base.output_values>` attribute. It is also used as the input to any projections for
which the outputState is the `sender <Projection.Projection.sender>`.

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

# Standard OutputStates
OUTPUT_RESULT = {NAME: RESULT}

OUTPUT_MEAN = {NAME:MEAN,
               CALCULATE:lambda x: np.mean(x)}

OUTPUT_MEDIAN = {NAME:MEDIAN,
                   CALCULATE:lambda x: np.median(x)}

OUTPUT_STAND_DEVIATION = {NAME:STANDARD_DEV,
                          CALCULATE:lambda x: np.std(x)}

OUTPUT_VARIANCE = {NAME:VARIANCE,
                   CALCULATE:lambda x: np.var(x)}


# # This is a convenience class that provides list of names of standard_output_state names in IDE
# class OUTPUT_STATES():
#     RESULT=RESULT
#     MEAN=MEAN
#     MEDIAN=MEDIAN
#     STANDARD_DEV=STANDARD_DEV
#     VARIANCE=VARIANCE
#
#
# standard_output_states = [{NAME: RESULT},
#                           {NAME:MEAN,
#                            CALCULATE:lambda x: np.mean(x)},
#                           {NAME:MEDIAN,
#                            CALCULATE:lambda x: np.median(x)},
#                           {NAME:STANDARD_DEV,
#                            CALCULATE:lambda x: np.std(x)},
#                           {NAME:VARIANCE,
#                            CALCULATE:lambda x: np.var(x)}]
#
#


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

    Implements a subclass of `State` that represents an output of a mechanism.

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
        the `mechanism <Mechanism>` to which the outputState belongs; it must be specified or determinable from the
        context in which the outputState is created.

    reference_value : number, list or np.ndarray
        a template that specifies the format of the item of the owner mechanism's
        `value <Mechanism.Mechanism_Base.value>` attribute to which the outputState will be assigned (specified by
        the **index** argument).  This must match (in number and type of elements) the outputState's
        **variable** argument.  It is used to insure the compatibility of the source of the
        input for the outputState with its `variable <OutputState.variable>`.

    value : number, list or np.ndarray
        specifies the template for the outputState's `value <OutputState.value>`.

    index : int : default PRIMARY_OUTPUT_STATE
        specifies the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` used as input for the
        function specified by the outputState's `calculate <OutputState.calculate>` attribute, to determine the
        outputState's `value <OutputState.value>`.

    calculate : Function, function, or method : default Linear
        specifies the function used to convert the designated item of the owner mechanism's
        `value <Mechanism.Mechanism_Base.value>` (specified by the outputState's :keyword:`index` attribute),
        before it is assigned as the outputState's `value <OutputState.value>`.  The function must accept a value that
        has the same format (number and type of elements) as the item of the mechanism's
        `value <Mechanism.Mechanism_Base.value>`.

    function : Function, function, or method : default LinearCombination(operation=SUM)
        function used to aggregate the values of the projections received by the outputState.
        It must produce a result that has the same format (number and type of elements) as the item of the mechanism's
        `value <Mechanism.Mechanism_Base.value>` to which the outputState is assigned (specified by its
        **index** argument).

        .. note::
             This is not used a present (see `note <OutputState_Function_Note_2>` for additonal details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the outputState, its function, and/or a custom function and its parameters. Values specified for parameters
        in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the outputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the outputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the outputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism
        the mechanism to which the outputState belongs.

    sendsToProjections : Optional[List[Projection]]
        a list of the projections sent by the outputState (i.e., for which the outputState is a
        `sender <Projection.Projection.sender>`).

    variable : number, list or np.ndarray
        assigned the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` specified by the
        outputState's `index <OutputState.index>` attribute.

    index : int
        the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` used as input for the function
        specified by its `calculate <OutputState.calculate>` attribute.

    calculate : function or method : default Linear
        function used to convert the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` specified by
        the outputState's `index <OutputState.index>` attribute.  The result is combined with the result of the
        outputState's `function <OutputState.function>` ((though see `note below <OutputState_Function_Note_2>`)
        to determine both the `value <OutputState.value>` of the outputState, as well as the value of the
        corresponding item of the owner mechanism's `output_values <Mechanism.Mechanism_Base.output_values>`.
        The default (`Linear`) transfers the value unmodified.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation  of the values of the projections received by the
        outputState.  The result is combined with the result of the function specified by
        `calculate <OutputState.calculate>`, and assigned as both the outputState's `value <OutputState.value>`
        and the corresponding item of the owner's `output_values <Mechanism.Mechanism_Base.output_values>`.

        .. _OutputState_Function_Note_2:

        .. note::
           PsyNeuLink does not currently support projections to outputStates.  Therefore, the
           :keyword:`function` attribute is not used.  It is implemented strictly for consistency with other
           state classes, and for potential future use.
           COMMENT:
             and for potential future use.  The default simply
             passes its input to its output. The :keyword:`function` attribute can be modified to change this behavior.
             However, to insure compatibility with future versions, it is *strongly* recommended that such functionality
             be implemented by assigning the desired function to the `calculate <OutputState.calculate>` attribute.
           COMMENT

    value : number, list or np.ndarray
        assigned the result of `function <OutputState.function>`
        (though see note under `function <OutputState.function>) combined with the result of the function specified
        by `calculate <OutputState.calculate>`;  the same value is assigned to the corresponding item of the owner
        mechanism's `output_values <Mechanism.Mechanism_Base.output_values>`.

    name : str : default <State subclass>-<index>
        name of the outputState.
        Specified in the **name** argument of the constructor for the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the outputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

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
        """Validate index and calculate parameters

        Validate that index is within the range of the number of items in the owner mechanism's ``value``,
        and that the corresponding item is a valid input to the calculate function


        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if INDEX in target_set:
            try:
                self.owner.value[target_set[INDEX]]
            except IndexError:
                raise OutputStateError("Value of {} argument for {} is greater than the number of items in "
                                       "the output_values ({}) for its owner mechanism ({})".
                                       format(INDEX, self.name, self.owner.output_values, self.owner.name))

        # IMPLEMENT: VALIDATE THAT CALCULATE FUNCTION ACCEPTS VALUE CONSISTENT WITH
        #            CORRESPONDING ITEM OF OWNER MECHANISM'S VALUE
        if CALCULATE in target_set:
            try:
                if isinstance(target_set[CALCULATE], type):
                    function = target_set[CALCULATE]().function
                else:
                    function = target_set[CALCULATE]
                try:
                    function(self.owner.value[target_set[INDEX]])
                except:
                    raise OutputStateError("Item {} of value for {} ({}) is not compatible with the function "
                                           "specified for the {} parameter of {} ({})".
                                           format(target_set[INDEX],
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

        # FIX: FOR NOW, self.value IS ALWAYS None (SINCE OUTPUTSTATES DON'T GET PROJECTIONS, AND
        # FIX:     AND State.update RETURNS None IF THERE ARE NO PROJECTIONS, SO IT ALWAYS USES CALCULATE (BELOW).
        # FIX:     HOWEVER, NEED TO INTEGRATE self.value and self.function WITH calculate:
        # IMPLEMENT: INCORPORATE paramModulationOperation HERE, AS PER PARAMETER STATE:
        #            TO COMBINE self.value ASSIGNED IN CALL TO SUPER (FROM PROJECTIONS)
        #            WITH calculate(self.owner.value[index]) PER BELOW

        # MODIFIED 5/11/17 OLD:
        self.value = type_match(self.calculate(self.owner.value[self.index]), type(self.owner.value[self.index]))
        # # MODIFIED 5/11/17 NEW:
        # calculated_value = type_match(self.calculate(self.owner.value[self.index]), type(self.owner.value[self.index]))
        # self.value = self.function(calculated_value)
        # MODIFIED 5/11/17 END




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

    Create ContentAddressableList of outputState(s) specified in paramsCurrent[OUTPUT_STATES]
    If OUTPUT_STATES is not specified, use self.value to create a default output state
    When completed:
        - self.outputStates contains a ContentAddressableList of one or more outputStates;
        - self.output_state contains first or only outputState in list;
        - paramsCurrent[OUTPUT_STATES] contains the same ContentAddressableList (of one or more outputStates)
        - each outputState corresponds to an item in the output of the owner's function
        - if there is only one outputState, it is assigned the full value

    (See State._instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[OUTPUT_STATES] (self.value) are assigned here
        rather than in _validate_params, as it requires function to have been instantiated first
    """

    constraint_value = []

    # Get owner.value
    # IMPLEMENTATION NOTE:  ?? IS THIS REDUNDANT WITH SAME TEST IN Mechanism.execute ?  JUST USE RETURN VALUE??
    owner_value = owner.value
    # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
    #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
    #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
    if (isinstance(owner_value, list) and
        (all(isinstance(item, np.ndarray) for item in owner_value) and
            all(
                    all(item.shape[i]==owner_value[0].shape[0]
                        for i in range(len(item.shape)))
                    for item in owner_value))):
        pass
    else:
        converted_to_2d = np.atleast_2d(owner.value)
        # If owner_value is a list of heterogenous elements, use as is
        if converted_to_2d.dtype == object:
            owner_value = owner.value
        # Otherwise, use value converted to 2d np.array
        else:
            owner_value = converted_to_2d

    # Get the value of each outputState
    # IMPLEMENTATION NOTE:
    # Should change the default behavior such that, if len(owner_value) == len owner.paramsCurrent[OUTPUT_STATES]
    #        (that is, there is the same number of items in owner_value as there are outputStates)
    #        then increment index so as to assign each item of owner_value to each outputState
    if owner.output_states:
        for output_state in owner.output_states:

            # Default is PRIMARY_OUTPUT_STATE
            index = PRIMARY_OUTPUT_STATE
            output_state_value = owner_value[index]

            # output_state is:

            # OutputState object, so get its index attribute
            if isinstance(output_state, OutputState):
                index = output_state.index
                output_state_value = owner_value[index]

            # string, so check if it is the name of a standard_output_state and, if so, get its dict
            elif isinstance(output_state, str) and hasattr(owner, STANDARD_OUTPUT_STATES):
                # check if string matches the name entry of a dict in standard_output_states
                item = next((item for item in owner.standard_output_states.names if output_state is item), None)
                if item is not None:
                    # assign dict to owner's output_state list
                    owner.output_states[owner.output_states.index(output_state)] = \
                                                                owner.standard_output_states.get_dict(output_state)
                    # re-assign output_state to dict so it is processed below
                    output_state = item

            # specification dict, so get its INDEX attribute if specified, and apply calculate function if specified
            if isinstance(output_state, dict):
                try:
                    index = output_state[INDEX]
                except KeyError:
                    pass
                if CALCULATE in output_state:
                    output_state_value = output_state[CALCULATE](owner_value[index])
                else:
                    output_state_value = owner_value[index]

            constraint_value.append(output_state_value)
    else:
        constraint_value = owner_value

    state_list = _instantiate_state_list(owner=owner,
                                         state_list=owner.output_states,
                                         state_type=OutputState,
                                         state_param_identifier=OUTPUT_STATES,
                                         constraint_value=constraint_value,
                                         constraint_value_name="output",
                                         context=context)

    # FIX: This is a hack to avoid recursive calls to assign_params, in which output_states never gets assigned
    # FIX: Hack to prevent recursion in calls to setter and assign_params
    if 'COMMAND_LINE' in context:
        owner.output_states = state_list
    else:
        owner._output_states = state_list


class StandardOutputStates():
    """Assign names and indices of specification dicts in standard_output_state as properties of the owner's class 
    
    Provide access to dicts and lists of the values of their name and index entries. 
    
    """

    @tc.typecheck
    def __init__(self, owner:Component, output_state_dicts:list):

        for item in output_state_dicts:
            if not isinstance(item, dict):
                raise OutputStateError("All items of {} for {} must be dicts ({} is not)".
                                     format(self.__class__.__name__, owner.componentName, item))
        self.data = output_state_dicts

        for i, state_dict in enumerate(self.data):
            state_dict[INDEX] = i

        # Add names of each outputState as property of the owner's class that returns its name string
        for state in self.data:
            setattr(owner.__class__, state[NAME], make_readonly_property(state[NAME]))

        # Add <NAME_INDEX> of each outputState as property of the owner's class that returns its index
        for state in self.data:
            setattr(owner.__class__, state[NAME]+'_INDEX', make_readonly_property(state[INDEX]))

    @tc.typecheck
    def get_dict(self, name:str):
        return self.data[self.names.index(name)]
    
    @property
    def names(self):
        return [item[NAME] for item in self.data]

    @property
    def indices(self):
        return [item[INDEX] for item in self.data]



def make_readonly_property(val):
    """Return property that provides read-only access to its value
    """

    def getter(self):
        return val

    def setter(self, val):
        raise UtilitiesError("{} is read-only property of {}".format(val, self.__class__.__name__))

    # Create the property
    prop = property(getter).setter(setter)
    return prop
