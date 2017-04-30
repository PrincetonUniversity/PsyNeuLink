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
(referenced by the mechanism's `matrix <RecurrentTransferMechanism.matrix>` parameter).
  
.. _Recurrent_Transfer_Creation:

Creating a RecurrentTransferMechanism
-------------------------------------

A RecurrentTransferMechanism can be created directly by calling its constructor, or using the 
`mechanism() <Mechanism.mechanism>` function and specifying RECURRENT_TRANSFER_MECHANISM as its 
**mech_spec** argument.  The recurrent projection is created using the **matrix** argument of the mechanism's 
constructor, which must specify either a square matrix or a `MappingProjection` that uses one (the default is 
`FULL_CONNECTIVITY_MATRIX`).  In all other respects, a RecurrentTransferMechanism is specified in the same way as a 
standard `TransferMechanism`.

.. _Recurrent_Transfer_Structure:

Structure
---------

The distinguishing feature of a RecurrentTransferMechanism is its `matrix <RecurrentTransferMechanism.matrix>` 
parameter, which specifies a self-projecting MappingProjection;  that is, one that projects from the mechanism's 
`primary outputState <OutputState_Primary>` back to it `primary inputState <Mechanism_InputStates>`.  
In all other respects the mechanism is identical to a standard `TransferMechanism`. 

.. _Recurrent_Transfer_Execution:

Execution
---------

When a RecurrentTransferMechanism executes, it includes in its input the value of its 
`primary outputState <OutputState_Primary>` from the last :ref:`round of execution <LINK>`.

Like a `TransferMechanism`, the function used to update each element can be assigned using its
`function <TransferMechanism.function>` parameter.  When a RecurrentTransferMechanism is executed, it transforms its 
input (including from the recurrent projection) using the specified function and parameters (see 
`Transfer_Execution`), and returns the results in its outputStates.

COMMENT

.. _Recurrent_Transfer_Class_Reference:

Class Reference
---------------


"""

from PsyNeuLink.Components.Functions.Function import get_matrix, is_matrix, Energy
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection


class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


RECURRENT_ENERGY = "energy"
RECURRENT_ENTROPY = "entropy"


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
            RecurrentTransferMechanism is a Subtype of the TransferMechanism Subtype of the ProcessingMechanisms Type 
            of the Mechanism Category of the Component class.
            It implements a TransferMechanism with a recurrent projection (default matrix: FULL_CONNECTIVITY_MATRIX).
            In all other respects, it is identical to a TransferMechanism 
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

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent MappingProjection <Recurrent_Transfer_Structure>`, 
        or a MappingProjection to use. 

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
    instance of RecurrentTransferMechanism : RecurrentTransferMechanism


    Attributes
    ----------

    variable : value
        the input to mechanism's ``function``.  :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    function : Function
        the function used to transform the input.

    matrix : 2d np.array
        the `matrix <MappingProjection.matrix>` parameter of the `recurrent_projection` for the mechanism.

    recurrent_projection : MappingProjection
        a `MappingProjection` that projects from the mechanism's `primary outputState <OutputState_Primary>` 
        back to it `primary inputState <Mechanism_InputStates>`.

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

    time_constant : float
        the time constant for exponential time averaging of input
        when the mechanism is executed using the `TIME_STEP` `TimeScale`::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Tuple[float, float]
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

    time_scale :  TimeScale
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default TransferMechanism-<index>
        the name of the mechanism.
        Specified in the **name** argument of the constructor for the projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the **prefs** argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentType = RECURRENT_TRANSFER_MECHANISM

    paramClassDefaults = TransferMechanism.paramClassDefaults.copy()
    paramClassDefaults[OUTPUT_STATES].append({NAME:RECURRENT_ENERGY})
    paramClassDefaults[OUTPUT_STATES].append({NAME:RECURRENT_ENTROPY})


    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 function=Linear,
                 matrix:tc.any(is_matrix, MappingProjection)=FULL_CONNECTIVITY_MATRIX,
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

            matrix_param = target_set[MATRIX]
            size = len(self.variable[0])

            if isinstance(matrix_param, MappingProjection):
                matrix = matrix_param.matrix

            elif isinstance(matrix_param, str):
                matrix = get_matrix(matrix_param, size, size)

            else:
                matrix = matrix_param

            if matrix.shape[0] != matrix.shape[0]:
                if (matrix_param, MappingProjection):
                    if __name__ == '__main__':
                        err_msg = ("{} param of {} must be square to be used as recurrent projection for {}".
                                   format(MATRIX, matrix_param.name, self.name))
                else:
                    err_msg = "{} param for must be square".format(MATRIX, self.name)
                raise RecurrentTransferError(err_msg)

    def _instantiate_attributes_after_function(self, context=None):

        super()._instantiate_attributes_after_function(context=context)

        if isinstance(self.matrix, MappingProjection):
            self.recurrent_projection = self.matrix

        else:
            self.recurrent_projection = _instantiate_recurrent_projection(self, self.matrix)

        self.matrix = self.recurrent_projection.matrix

        self.outputStates[RECURRENT_ENERGY].calculate = Energy(self.variable,
                                                           self.recurrent_projection.parameterStates[MATRIX]).function

        TEST_CONDTION = True



# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
@tc.typecheck
def _instantiate_recurrent_projection(mech:Mechanism_Base,
                                      matrix:is_matrix=FULL_CONNECTIVITY_MATRIX):
    """Instantiate a MappingProjection from mech to itself

    """

    if isinstance(matrix, str):
        size = len(mech.variable[0])

    matrix = get_matrix(matrix, size, size)

    return MappingProjection(sender=mech,
                             receiver=mech,
                             matrix=matrix,
                             name = mech.name + ' recurrent projection')
