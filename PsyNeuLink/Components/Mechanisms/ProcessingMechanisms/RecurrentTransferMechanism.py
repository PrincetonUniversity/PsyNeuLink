# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND TIME_CONSTANT ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ****************************************  RecurrentTransferMechanism *************************************************

"""

Overview
--------

A RecurrentTransferMechanism is a subclass of TransferMechanism that implements a single-layered recurrent 
network, in which each element is connected to every other element by way of a recurrent MappingProjection
(referenced by the mechanism's ` matrix <RecurrentTransferMechanism.matrix>` parameter).
  
.. _Recurrent_Transfer_Creation:

Creating a RecurrentTransferMechanism
-------------------------------------

A RecurrentTransferMechanism can be created directly by calling its constructor, or using the :py:func:`mechanism`
function and specifying `RECURRENT_TRANSFER_MECHANISM` as its `mech_spec` argument.  The set of recurrent connections
are created by creating a MappingProjection of the type specified in the **matrix** argument of the mechanism's 
constructor.  In all other respects, it specified as a standard `TransferMechanism`.

.. _Recurrent_Transfer_Structure:

Structure
---------

The `matrix <Recurrent.matrix>` parameter of RecurrentTransferMechanism is a self-projecting MappingProjection;
that is, it projects from the mechanism's `primary outputState <OutputState_Primary>` to its
`primary inputState <Mechanism_InputStates>`.  In all other respects the mechanism is structured as a standard
`TransferMechanism`. 

.. _Transfer_Execution:

Execution
---------

When a RecurrentTransferMechanism executes, it includes in its input the value of its 
`primary outputState <OutputState_Primary>` from the last :ref:`round of execution <LINK>`.

Like a `TransferMechanism`, the function used to update each element can be assigned using its
`function <TransferMechanism.function>` parameter.  When a TransferMechanism is executed, it transforms its input 
using the specified function and the following parameters (in addition to those specified for the function):

    * `noise <TransferMechanism.noise>`: applied element-wise to the input before transforming it.
    ..
    * `time_constant <TransferMechanism.time_constant>`: if `time_scale` is :keyword:`TimeScale.TIME_STEP`,
      the input is exponentially time-averaged before transforming it (higher value specifies faster rate);
      if `time_scale` is :keyword:`TimeScale.TRIAL`, `time_constant <TransferMechanism.time_constant>` is ignored.
    ..
    * `range <TransferMechanism.range>`: caps all elements of the `function <TransferMechanism.function>` result by
      the lower and upper values specified by range.

The rate at which the network settles determined jointly by the size of the matrix and the time_constant
??Must use SOFT_CLAMP SO THAT THE INPUT DOESN'T DOMINATE THE RESULT??  

After each execution of the mechanism:

.. _Transfer_Results:

    * **result** of `function <TransferMechanism.function>` is assigned to the mechanism's
      `value <TransferMechanism.value>` attribute, the :keyword:`value` of its `TRANSFER_RESULT` outputState,
      and to the 1st item of the mechanism's `outputValue <TransferMechanism.outputValue>` attribute;
    ..
    * **mean** of the result is assigned to the the :keyword:`value` of the mechanism's `TRANSFER_MEAN` outputState,
      and to the 2nd item of its `outputValue <TransferMechanism.outputValue>` attribute;
    ..
    * **variance** of the result is assigned to the :keyword:`value` of the mechanism's `TRANSFER_VARIANCE` outputState,
      and to the 3rd item of its `outputValue <TransferMechanism.outputValue>` attribute.

COMMENT

.. _Transfer_Class_Reference:

Class Reference
---------------


"""

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import get_matrix, matrix_spec


class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class RecurrentTransferMechanism(TransferMechanism):
    """
    RecurrentTransferMechanism(        \
    default_input_value=None,          \
    function=Linear,                   \
    matrix=FULL_CONNECTIVITY_MATRIX,   \
    initial_value=None,                \
    noise=0.0,                         \
    time_constant=1.0,                 \
    range=(float:min, float:max),      \
    time_scale=TimeScale.TRIAL,        \
    params=None,                       \
    name=None,                         \
    prefs=None)

    Implements TransferMechanism subclass of `Mechanism`.

    COMMENT:
        Description
        -----------
            TransferMechanism is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the
                Component class
            It implements a Mechanism that transforms its input variable based on FUNCTION (default: Linear)

        Class attributes
        ----------------
            + componentType (str): TransferMechanism
            + classPreference (PreferenceSet): Transfer_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (value):  Transfer_DEFAULT_BIAS
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL}
            + paramNames (dict): names as above

        Class methods
        -------------
            None

        MechanismRegistry
        -----------------
            All instances of TransferMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_input_value : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary outputState <OutputState_Primary>`
        of the mechanism.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `time_constant <TransferMechanism.time_constant>` is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the mechanism is executed with `time_scale`
        set to `TimeScale.TIME_STEP`::

         result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <TransferMechanism.range>` that it exceeds.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.
        This must be set to `TimeScale.TIME_STEP` for the `time_constant <TransferMechanism.time_constant>`
        parameter to have an effect.

    name : str : default TransferMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    .. context=componentType+INITIALIZING):
            context : str : default ''None''
                   string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Returns
    -------
    instance of TransferMechanism : TransferMechanism


    Attributes
    ----------

    variable : value: default Transfer_DEFAULT_BIAS
        the input to mechanism's ``function``.  :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    function : Function :  default Linear
        the function used to transform the input.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `time_constant <TransferMechanism.time_constant>` parameter is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <TransferMechahnism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input
        when the mechanism is executed using the `TIME_STEP` `TimeScale`::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Optional[Tuple[float, float]]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <TransferMechanism.range>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `range <TransferMechanism.range>` is set by default to (0,1).

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`; same value as fist item of
        `outputValue <TransferMechanism.outputValue>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the :keyword:`TRANSFER_RESULT` outputState
            and the first item of ``outputValue``.
    COMMENT

    outputStates : Dict[str, OutputState]
        an OrderedDict with three `outputStates <OutputState>`:
        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <TransferMechanism.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;

    outputValue : List[array(float64), float, float]
        a list with the following items:
        * **result** of the ``function`` calculation (value of `TRANSFER_RESULT` outputState);
        * **mean** of the result (``value`` of `TRANSFER_MEAN` outputState)
        * **variance** of the result (``value`` of `TRANSFER_VARIANCE` outputState)

    time_scale :  TimeScale : defaul tTimeScale.TRIAL
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default TransferMechanism-<index>
        the name of the mechanism.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the `prefs` argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = RECURRENT_TRANSFER_MECHANISM


    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 function=Linear,
                 matrix:matrix_spec=FULL_CONNECTIVITY_MATRIX,
                 initial_value=None,
                 noise=0.0,
                 time_constant=1.0,
                 range=None,
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Assign type-level preferences, default input value (Transfer_DEFAULT_BIAS) and call super.__init__
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix)

        super().__init__(
                 default_input_value=default_input_value,
                 function=function,
                 initial_value=initial_value,
                 noise=noise,
                 time_constant=time_constant,
                 range=range,
                 time_scale=time_scale,
                 params=params,
                 name=name,
                 prefs=prefs,
                 context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate MATRIX
        if MATRIX in target_set:
            matrix = target_set[MATRIX]
            size = len(self.variable[0])
            if isinstance(matrix, str):
                matrix = get_matrix(matrix, size, size)
            if matrix.shape[0] != matrix.shape[0]:
                raise RecurrentTransferError("{} param for {} must be square".format(MATRIX, self.name))

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        if isinstance(self.matrix, str):
            size = len(self.variable[0])
            self.matrix = get_matrix(self.matrix, size, size)

        self.matrix = _instantiate_recurrent_projection(self, self.matrix)


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
@tc.typecheck
def _instantiate_recurrent_projection(mech:Mechanism_Base,
                                      matrix:matrix_spec=FULL_CONNECTIVITY_MATRIX):
    """Instantiate a MappingProjection from mech to itself

    """

    matrix = get_matrix(matrix)

    return MappingProjection(sender=mech,
                             receiver=mech,
                             matrix=matrix,
                             name = mech.name + ' recurrent projection')
