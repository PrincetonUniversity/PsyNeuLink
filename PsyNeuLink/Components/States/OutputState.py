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
mechanism's `output_value <Mechanism.Mechanism_Base.output_value>` attribute.  OutputStates are used to represent 
individual items of the mechanism's `value <Mechanism.Mechanism_Base.value>`, and/or useful quantities derived from 
them.  For example, the `function <TransferMechanism.TransferMechanism.function>` of a `TransferMechanism` generates 
a single result (the transformed value of its input);  however, a TransferMechanism can also be assigned outputStates 
that represent its mean, variance or other derived values.  In contrast, the `function <DDM.DDM.function>` 
of a `DDM` mechanism generates several results (such as decision accuracy and response time), each of which can be 
assigned as the `value <OutputState.value>` of a different outputState.  The outputState(s) of a mechanism can serve 
as the input to other  mechanisms (by way of `projections <Projections>`), or as the output of a process and/or 
system.  The outputState's `efferents <OutputState.efferents>` attribute lists all of its outgoing 
projections.

.. _OutputStates_Creation:

Creating an OutputState
-----------------------

An outputState can be created by calling its constructor. However, in general this is not necessary, as a mechanism
automatically creates a default outputState if none is explicitly specified, that contains the primary result of its
`function <Mechanism_Base.function>`.  For example, if the mechanism is created within the `pathway` of a 
`process <Process>`, an outputState will be created and assigned as the
`sender <MappingProjection.MappingProjection.sender>` of a `MappingProjection` to the next mechanism in the pathway,
or to the process's `output <Process_Input_And_Output>` if the mechanism is a `TERMINAL` mechanism for that process.
Process_Input_And_Ouput.  Other configurations can also easily be specified using a mechanism's 
`output_states <Mechanism_Base.output_states>` attribute (see `OutputState_Specification` below).

An outputState must be owned by a mechanism. Therefore, if the outputState is created explicitly, the mechanism to
which it belongs must be specified in the **owner** argument of its constructor; if the outputState is specified
in the OUTPUT_STATES entry of the `parameter dictionary <ParameterState_Specifying_Parameters>` for a
mechanism, then the owner is inferred from the context.

.. _OutputState_Primary:

Primary OutputState
~~~~~~~~~~~~~~~~~~~

Every mechanism has at least one outputState, referred to as its *primary outputState*.  If outputStates are not
`explicitly specified <OutputState_Specification>` for a mechanism, a primary outputState is automatically created
and assigned to its `outputState <Mechanism.Mechanism_Base.outputState>` attribute (note the singular),
and also to the first entry of the mechanism's `outputStates <Mechanism.Mechanism_Base.outputStates>` attribute 
(note the plural).  The `value <OutputState.value>` of the primary outputState is assigned as the first (and often 
only) item of the mechanism's `output_value <Mechanism.Mechanism_Base.output_value>`, which is the result of the 
mechanism`s `function <Mechanism.Mechanism_Base.function>`.

.. _OutputState_Specification:

OutputState Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

The primary outputState of a mechanism can be supplemented or replaced using the **output_states** argument of a 
mechanism's constructor <Mechanism_OutputStates>`.  If any are specified there, *all*  of the outputStates desired 
must be specified there;  that is, explicitly specifying *any* outputStates in an **output_states** argument 
replaces any defaults.  Therefore, if the default outputState -- that usually contains the result of the mechanism's
function -- is to be retained, it too must be specified along with any additional ones desired.  OutputStates are 
specified in a list, either in the **output_states** argument of a `mechanism's constructor <Mechanism_Creation>`, or
by direct assignment to its `output_states <OutputState.output_states>` attribute.  The specification of each 
outputState in a list can be any of the following: 

    * A reference to an **existing outputState**.  Its `variable <OutputState.variable>` must match (in the
      number and type of its elements) the item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` to
      which the outputState is assigned (designated by its `index <OutputState_Index>` attribute).
    ..
    COMMENT:
    * A reference to an **existing mechanism**.    
    ..
    COMMENT
    * A **specification dictionary**.  Each entry should use the name of an argument for
      an outputState parameter (used in the outputState constructor) as its key, and the value of that
      parameter as its value.  By default, the outputState is assigned to the first item of the owner mechanism's
      `value <Mechanism.Mechanism_Base.value>`.  However, an entry with :keyword:`INDEX <OutputState_Index>` as its
      key can be used to assign it to different item of the mechanism's `value <Mechanism.Mechanism_Base.value>`.
    ..
    * A **string**.  This creates a default outputState using that string as its name, and is assigned as the
      `primary outputState <OutputState_Primary>` for the mechanism.
    ..
    * A **value**.  This creates a default outputState using the specified value as the outputState's
      `variable <OutputState.value>`.  This must be compatible with (have the same number and type of elements as) the
      item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` to which the outputState will be assigned
      (its primary outputState by default, or designated by its `index <OutputState.index>` attribute).  A default 
      name will be assigned based on the name of the mechanism (see :ref:`naming conventions <LINK>`).
      
      COMMENT:
         AT PRESENT THIS IS NOT USEFUL;  HOWEVER, IN THE FUTURE (E.G., WHEN GATING PROJECTIONS TO OUTPUT STATES
         IS ADDED) IT MAY BE USEFUL FOR SPECIFYING A BASEVALUE (E.G., DEFAULT) FOR THE OUTPUTSTATE.
      COMMENT

    .. note::
       In all cases, the `variable <OutputState.variable>` of the outputState must match (have the same number and
       type of elements) as the item of its owner mechanism's `value <Mechanism.Mechanism_Base.value>` to which it is
       assigned.



The list of outputStates for a mechanism can be accessed using its `output_states <Mechanism_Base.output_states>` 
attribute, and new ones can be added to an existing mechanism by appending a list (containing specifications in the
form listed above) to the output_states list. Assigning without appending will replace any existing outputStates.  
If the name of an explicitly specified outputState is the same as one that was created automatically (or another one 
that was created explicitly), its name will be suffixed with a numerical index (incremented for each outputState with 
that name), and the outputState will be added to the list (that is, it will *not* replace ones that were already 
created).

The values of a mechanism's outputStats will be assigned as items in its `output_value <Mechanism.output_value>` 
attribute, in the order in which they are assigned in the **output_states** argument of its constructor, and listed
in its `output_states <Mechanism.output_states>` attribute.  Note that the `output_value <Mechanism.output_value>`
of a mechanism is distinct from its `value <Mechanism_Base.value>` attribute, which contains the full and unmodified 
results of its `function <Mechanism_Base.function>`.

.. _OutputState_Standard:

Standard OutputStates
^^^^^^^^^^^^^^^^^^^^^  

Most types of mechanisms have a `standard_output_states` class attribute, that contains a list of predefined 
outputStates relevant to that type of mechanism (for example, the TransferMechanism class has outputStates for 
calculating the mean, median, variance, and standard deviation of its result).  The names of these are listed as 
attributes of a class with the name <ABBREVIATED_CLASS_NAME>_OUTPUT>.  For example, the TransferMechanism class 
defines `TRANSFER_OUTPUT`, with attributes MEAN, MEDIAN, VARIANCE and STANDARD_DEV that are the names of 
predefined outputStates in its `standard_output_states <TransferMechanism.standard_output_states>` attribute.
These can be used in the list of outputStates specified for a TransferMechanism object, as in the example below:: 

    my_mech = TransferMechanism[default_input_value=[0,0],
                                function=Logistic(),
                                output_states=[TRANSFER_OUTPUT.RESULT, 
                                               TRANSFER_OUTPUT.MEAN,
                                               TRANSFER_OUTPUT.VARIANCE]

In this example, ``my_mech`` is configured with three outputStates;  the first will be named *RESULT* and will 
represent logistic transform of the 2-element input vector;  the second will be named  *MEAN* and will represent mean 
of the result (i.e., of its two elements); and the third will be named *VARIANCE* and contain the variance of the 
result.

.. _OutputState_Customization:

OutputState Customization
~~~~~~~~~~~~~~~~~~~~~~~~~

The default outputState uses the first (and usually only) item of the owner mechanism's 
`value <Mechanism.Mechanism_Base.value>` as its value.  However, this can be modified in two ways, using the
outputState's `index <OutputState.index>` and `calculate <OutputState.calculate>` attributes (see 
`OutputState_Attributes` below). If the mechanism's `function <Mechanism.function>` returns a value with more than one 
item (i.e., a list of lists, or a 2d np.array), then an outputState can be assigned to any of those items by specifying 
its `index <OutputState.index>` attribute. An outputState can also be configured to transform
the value of the item, by specifying a function for its `calculate <OutputState.calculate>` attribute; the result
will then be assigned as the outputState's `value <OutputState.value>`.  An outputState's `index <OutputState.index>` 
and `calculate <OutputState.calculate>` attributes can be assigned when the outputState is assigned to a mechanism, 
by including *INDEX* and *CALCULATE* entries in a  `specification dictionary <OutputState_Specification>` for the 
outputState, as in the following example::

    my_mech = DDM(function=BogaczEtAl(),
                  output_states=[ DDM.DECISION_VARIABLE,
                                  DDM.PROB_UPPER_THRESHOLD,
                                  { NAME: 'DECISION ENTROPY',
                                    INDEX: 2},
                                    CALCULATE: Entropy().function } ] )

COMMENT:
   ADD VERSION IN WHICH INDEX IS SPECIFICED USING DDM_standard_output_states
COMMENT

In this example, ``my_mech`` is configured with three outputStates.  The first two are standard outputStates that 
represent the decision variable of the DDM and the probability of it crossing of the upper (vs. lower) threshold.  the 
third is a custom outputState, that computes the entropy of the probability of crossing the upper threshold.  It uses
the `Entropy` Function for its `calculate <OutputState.calculate>` attribute, and INDEX is assigned ``2`` to reference  
the third item of the DDM's `value <DDM.value>` attribute, which contains the probability of crossing the upper 
threshold.  The three outputStates will be assigned to the `output_states <Mechanism.output_states>` attribute of 
``my_mech``, and their values will be assigned as items in its `output_value <Mechanism.output_value>` attribute, 
each in the order in which it is listed in the **output_states** argument of the constructor for ``my_mech``.  

.. _OutputState_Structure:

Structure
---------

Every outputState is owned by a `mechanism <Mechanism>`. It can send one or more
`MappingProjections <MappingProjection>` to other mechanisms.  If its owner is a `TERMINAL` mechanism of a process
and/or system, then the outputState will also be treated as the output of that `process <Process_Input_And_Output>`
and/or of a system.  The projections that the outputState sends are listed in its
`efferents <OutputState.efferents>` attribute.

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

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

An outputState also has two additional attributes that determine its operation, and can be used to further 
`customize the outputState <OutputState_Customization>`:

.. _OutputState_Index:

* `index <OutputState.index>`: this determines the item of its owner mechanism's
  `value <Mechanism.Mechanism_Base.value>` to which it is assigned.  By default, this is set to 0, which assigns it to
  the first item of the mechanism's `value <Mechanism.Mechanism_Base.value>`.  Its value must be equal to or less than
  one minus the number of outputStates in the mechanism's `output_states <Mechanism.output_states>` attribute.  In 
  addition, the `variable <OutputState.variable>` of an outputState must match (in the number and type of its elements) 
  the item of the mechanism's `value <Mechanism.Mechanism_Base.value>` to which the index refers.      

.. _OutputState_Calculate:

* `calculate <OutputState.calculate>`:  this specifies a function used to convert the item of the owner mechanism's
  `value <Mechanism.Mechanism_Base.value>` (designated by the outputState's `index <OutputState.index>` attribute),
  before assigning it as the outputState's `value <OutputState.vaue>`.  The result is used as the input to the
  outputState's `function <OutputState.function>` attribute (which implements the effects of any 
  `GatingProjections` to the outputState), the result of whch is assigned as the outputState's 
  `value <OutputState.value>` (though see `note <OutputState_Function_Note_1>`). The 
  `calculate  <OutputState.calculate>` attribute can be assigned any function that can take as its input the 
  item of the owner mechanism's `value <Mechanism.Mechanism_Base.value>` designated by the outputState's
  `index <OutputState.index>` attribute, and the result of which can be used as the variable for the outputState's 
  `function <OutputState.function>`.  The default is an identity function (`Linear` with **slope**\ =1 and 
  **intercept**\ =0), that simply assigns the specified item of the mechanism's `value <Mechanism.Mechanism_Base.value>` 
  unmodified as the variable for outputState's `function <OutputState.function>`.  

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
SEQUENTIAL = 'SEQUENTIAL'

# Standard OutputStates
# OUTPUT_RESULT = {NAME: RESULT}
#
# OUTPUT_MEAN = {NAME:MEAN,
#                CALCULATE:lambda x: np.mean(x)}
#
# OUTPUT_MEDIAN = {NAME:MEDIAN,
#                    CALCULATE:lambda x: np.median(x)}
#
# OUTPUT_STAND_DEVIATION = {NAME:STANDARD_DEV,
#                           CALCULATE:lambda x: np.std(x)}
#
# OUTPUT_VARIANCE = {NAME:VARIANCE,
#                    CALCULATE:lambda x: np.var(x)}


# This is a convenience class that provides list of standard_output_state names in IDE
class OUTPUTS():
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEV=STANDARD_DEV
    VARIANCE=VARIANCE

standard_output_states = [{NAME: RESULT},
                          {NAME:MEAN,
                           CALCULATE:lambda x: np.mean(x)},
                          {NAME:MEDIAN,
                           CALCULATE:lambda x: np.median(x)},
                          {NAME:STANDARD_DEV,
                           CALCULATE:lambda x: np.std(x)},
                          {NAME:VARIANCE,
                           CALCULATE:lambda x: np.var(x)}]


class OutputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OutputState(State_Base):
    """
    OutputState(                               \
    owner,                                     \
    reference_value,                           \
    variable=None,                             \
    function=LinearCombination(operation=SUM), \
    index=PRIMARY_OUTPUT_STATE,                \
    calculate=Linear,                          \
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

    variable : number, list or np.ndarray
        specifies the template for the outputState's `variable <OutputState.variable>`.

    function : Function, function, or method : default LinearCombination(operation=SUM)
        function used to aggregate the values of the projections received by the outputState.
        It must produce a result that has the same format (number and type of elements) as the item of the mechanism's
        `value <Mechanism.Mechanism_Base.value>` to which the outputState is assigned (specified by its
        **index** argument).

        .. note::
             This is not used a present (see `note <OutputState_Function_Note_2>` for additonal details).

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

    efferents : Optional[List[Projection]]
        a list of the projections sent by the outputState (i.e., for which the outputState is a
        `sender <Projection.Projection.sender>`).

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

    @tc.typecheck
    def __init__(self,
                 owner,
                 reference_value,
                 variable=None,
                 function=LinearCombination(operation=SUM),
                 index=PRIMARY_OUTPUT_STATE,
                 calculate:is_function_type=Linear,
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
            raise OutputStateError("Variable ({}) of outputState for {} is not compatible with "
                                           "the output ({}) of its function".
                                           format(self.variable,
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

    def _execute(self, function_params, context):
        """Call self.function with owner's value as variable
        """

        # IMPLEMENTATION NOTE: OutputStates don't current receive TransmissiveProjections,
        #                      so there is no need to use their value (as do InputStates)
        value = self.function(variable=self.owner.value[self.index],
                                params=function_params,
                                context=context)

        return type_match(self.calculate(self.owner.value[self.index]), type(value))

    @property
    def trans_projections(self):
        return self.efferents

    @trans_projections.setter
    def trans_projections(self, assignment):
        self.efferents = assignment



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
    # IMPLEMENTATION NOTE:  SHOULD BE REFACTORED TO USE _parse_state_spec TO PARSE ouput_states arg
    if owner.output_states:
        for i, output_state in enumerate(owner.output_states):

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
                    output_state = item

            # specification dict, so get its INDEX attribute if specified, and apply calculate function if specified
            elif isinstance(output_state, dict):
                try:
                    index = output_state[INDEX]
                except KeyError:
                    pass
                if CALCULATE in output_state:
                    output_state_value = output_state[CALCULATE](owner_value[index])
                else:
                    output_state_value = owner_value[index]

            else:
                raise OutputState("PROGRAM ERROR: unrecognized item ({}) in output_states specification for {}".
                                  format(output_state, owner.name))

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
    """Collection of OutputState specification dictionaries for standard_output_states of a class
    
    data attribute contains dictionary of OutputState specification dictionaries
    indices contains a list of the default index for each OutputState specified
    names containes a list of the default name of each OutputState

    **indices** arg of constructore specifies how to assign the INDEX entry for each dict listed in output_state_dicts;
        the arg can be:
            * PRIMARY_OUTPUT_STATES keyword - assigns the INDEX for the owner's primary outputState to all indices;
            * SEQUENTIAL keyword - assigns sequentially incremented int to each INDEX entry
           * a list of ints - assigns each int to the corresponding entry in output_state_dicts
        if it is not specified, assigns `None` to each INDEX entry.

    `get_dict` method takes a name as its arg and returns a copy of the designated OutputState specification dictionary

    """

    @tc.typecheck
    def __init__(self,
                 owner:Component,
                 output_state_dicts:list,
                 indices:tc.optional(tc.any(int, str, list))=None):

        # Validate that all items in output_state_dicts are dicts
        for item in output_state_dicts:
            if not isinstance(item, dict):
                raise OutputStateError("All items of {} for {} must be dicts (but {} is not)".
                                     format(self.__class__.__name__, owner.componentName, item))
        self.data = output_state_dicts.copy()

        # Assign indices

        # List was provided, so check that:
        # - it has the appropriate number of items
        # - they are all ints
        # and then assign each int to the INDEX entry in the corresponding dict in output_state_dicts
        # outputState
        if isinstance(indices, list):
            if len(indices) != len(output_state_dicts):
                raise OutputStateError("Length of the list of indices provided to {} for {} ({}) "
                                       "must equal the number of outputStates dicts provided ({})"
                                       "length".format(self.__class__.__name__,
                                                       owner.name,
                                                       len(indices),
                                                       len(output_state_dicts)))

            if not all(isinstance(item, int) for item in indices):
                raise OutputStateError("All the items in the list of indices provided to {} for {} ({}) must be ints".
                                       format(self.__class__.__name__, self.name, owner.name, index))

            for index, state_dict in zip(indices, self.data):
                state_dict[INDEX] = index

        # Assign indices sequentially based on order of items in output_state_dicts arg
        elif indices is SEQUENTIAL:
            for index, state_dict in enumerate(self.data):
                state_dict[INDEX] = index

        # Assign PRIMARY_OUTPUT_STATE as INDEX for all outputStates in output_state_dicts
        elif indices is PRIMARY_OUTPUT_STATE:
            for state_dict in self.data:
                state_dict[INDEX] = PRIMARY_OUTPUT_STATE

        # No indices specification, so assign None to INDEX for all outputStates in output_state_dicts
        else:
            for state_dict in self.data:
                state_dict[INDEX] = None


        # Add names of each outputState as property of the owner's class that returns its name string
        for state in self.data:
            setattr(owner.__class__, state[NAME], make_readonly_property(state[NAME]))

        # Add <NAME_INDEX> of each outputState as property of the owner's class, that returns its index
        for state in self.data:
            setattr(owner.__class__, state[NAME]+'_INDEX', make_readonly_property(state[INDEX]))

    @tc.typecheck
    def get_dict(self, name:str):
        return self.data[self.names.index(name)].copy()
    
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
