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

A RecurrentTransferMechanism is a subclass of `TransferMechanism` that implements a single-layered recurrent 
network, in which each element is connected to every other element (instantiated in a recurrent MappingProjection
referenced by the Mechanism's `matrix <RecurrentTransferMechanism.matrix>` parameter).  It also allows its
previous input to be decayed, and reports the energy and, if appropriate, the entropy of its output.
  
.. _Recurrent_Transfer_Creation:

Creating a RecurrentTransferMechanism
-------------------------------------

A RecurrentTransferMechanism can be created directly by calling its constructor, or using the 
`mechanism() <Mechanism.mechanism>` function and specifying RECURRENT_TRANSFER_MECHANISM as its 
**mech_spec** argument.  The recurrent projection is created using the **matrix** argument of the Mechanism's
constructor, which must specify either a square matrix or a `MappingProjection` that uses one (the default is 
`FULL_CONNECTIVITY_MATRIX`).  In all other respects, a RecurrentTransferMechanism is specified in the same way as a 
standard `TransferMechanism`.

.. _Recurrent_Transfer_Structure:

Structure
---------

The distinguishing feature of a RecurrentTransferMechanism is its `matrix <RecurrentTransferMechanism.matrix>` 
parameter, which specifies a self-projecting MappingProjection;  that is, one that projects from the Mechanism's
`primary OutputState <OutputState_Primary>` back to it `primary InputState <Mechanism_InputStates>`.
In all other respects the Mechanism is identical to a standard `TransferMechanism`.

In addition, a RecurrentTransferMechanism also has a `decay` <RecurrentTransferMechanism.decay>' parameter, that
decrements its `previous_input <TransferMechanism.previous_input>` value by the specified factor each time it is
executed.  It also has two additional OutputStates:  an ENERGY OutputState and, if its
`function <TransferMechanisms.function>` is bounded between 0 and 1 (e.g., a `Logistic` function), an ENTROPY
OutputState, that each report the respective values of the vector in it its
`primary (RESULTS) OutputState <OutputState_Primary>`.
 
.. _Recurrent_Transfer_Execution:

Execution
---------

When a RecurrentTransferMechanism executes, it includes in its input the value of its 
`primary OutputState <OutputState_Primary>` from its last execution.

Like a `TransferMechanism`, the function used to update each element can be assigned using its
`function <TransferMechanism.function>` parameter.  When a RecurrentTransferMechanism is executed,
if its `decay <RecurrentTransferMechanism.decay>` parameter is specified (and is not 1.0), it 
decays the value of its `previous_input <TransferMechanism.previous_input>` parameter by the
specified factor.  It then transforms its input (including from the recurrent projection) using the specified 
function and parameters (see `Transfer_Execution`), and returns the results in its OutputStates.

.. _Recurrent_Transfer_Class_Reference:

Class Reference
---------------


"""

from PsyNeuLink.Components.Functions.Function import get_matrix, Stability
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.States.OutputState import StandardOutputStates, PRIMARY_OUTPUT_STATE


class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

DECAY = 'decay'

# This is a convenience class that provides list of standard_output_state names in IDE
class RECURRENT_OUTPUT():
    
    """
        .. _RecurrentTransferMechanism_Standard_OutputStates:

        `Standard OutputStates <OutputState_Standard>` for
        `RecurrentTransferMechanism`

        .. TRANSFER_RESULT:

        *RESULT* : 1d np.array
            the result of the `function <RecurrentTransferMechanism.function>`
            of the Mechanism

        .. TRANSFER_MEAN:

        *MEAN* : float
            the mean of the result

        *VARIANCE* : float
            the variance of the result

        .. ENERGY:

        *ENERGY* : float
            the energy of the result, which is calculated using the `Stability
            Function <Function.Stability.function>` with the ``ENERGY`` metric

        .. ENTROPY:

        *ENTROPY* : float
            The entropy of the result, which is calculated using the `Stability
            Function <Function.Stability.function>` with the ENTROPY metric
            (Note: this is only present if the Mechanism's `function` is bounded
            between 0 and 1 (e.g. the `Logistic` Function)).
        """
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE
    ENERGY=ENERGY
    ENTROPY=ENTROPY
    # THIS WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
    # for item in [item[NAME] for item in DDM_standard_output_states]:
    #     setattr(DDM_OUTPUT.__class__, item, item)


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class RecurrentTransferMechanism(TransferMechanism):
    """
    RecurrentTransferMechanism(        \
    default_input_value=None,          \
    size=None,                         \
    function=Linear,                   \
    matrix=FULL_CONNECTIVITY_MATRIX,   \
    initial_value=None,                \
    decay=None,                        \
    noise=0.0,                         \
    time_constant=1.0,                 \
    range=(float:min, float:max),      \
    time_scale=TimeScale.TRIAL,        \
    params=None,                       \
    name=None,                         \
    prefs=None)

    Implements RecurrentTransferMechanism subclass of `TransferMechanism`.

    COMMENT:
        Description
        -----------
            RecurrentTransferMechanism is a Subtype of the TransferMechanism Subtype of the ProcessingMechanisms Type 
            of the Mechanism Category of the Component class.
            It implements a TransferMechanism with a recurrent projection (default matrix: FULL_CONNECTIVITY_MATRIX).
            In all other respects, it is identical to a TransferMechanism.
    COMMENT

    Arguments
    ---------

    default_input_value : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary outputState <OutputState_Primary>`
        of the mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent MappingProjection <Recurrent_Transfer_Structure>`, 
        or a MappingProjection to use. 

    decay : number : default 1.0
        specifies the amount by which to decrement its `previous_input <TransferMechanism.previous_input>`
        each time it is executed.

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

         result = (time_constant * current input) +
         (1-time_constant * result on previous time_step)

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

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <RecurrentTransferMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <MappingProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : MappingProjection
        a `MappingProjection` that projects from the Mechanism's `primary outputState <OutputState_Primary>`
        back to it `primary inputState <Mechanism_InputStates>`.

    decay : float : default 1.0
        determines the amount by which to multiply the `previous_input <TransferMechanism.previous_input>` value
        each time it is executed.

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
        when the Mechanism is executed using the `TIME_STEP` `TimeScale`::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Tuple[float, float]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <TransferMechanism.range>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `range <TransferMechanism.range>` is set by default to (0,1).

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`; same value as first item of
        `output_values <TransferMechanism.output_values>`.    

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    outputStates : Dict[str, OutputState]
        an OrderedDict with the following `outputStates <OutputState>`:

        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <TransferMechanism.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;
        * `ENERGY`, the :keyword:`value` of which is the energy of the result, 
          calculated using the `Stability` Function with the ENERGY metric;
        * `ENTROPY`, the :keyword:`value` of which is the entropy of the result,
          calculated using the `Stability` Function with the ENTROPY metric; 
          note:  this is only present if the mechanism's :keyword:`function` is bounded between 0 and 1 
          (e.g., the `Logistic` function).

    output_values : List[array(float64), float, float]
        a list with the following items:

        * **result** of the ``function`` calculation (value of TRANSFER_RESULT outputState);
        * **mean** of the result (``value`` of TRANSFER_MEAN outputState)
        * **variance** of the result (``value`` of TRANSFER_VARIANCE outputState);
        * **energy** of the result (``value`` of ENERGY outputState);
        * **entropy** of the result (if the ENTROPY outputState is present).

    time_scale :  TimeScale
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default TransferMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    Returns
    -------
    instance of RecurrentTransferMechanism : RecurrentTransferMechanism

    """
    componentType = RECURRENT_TRANSFER_MECHANISM

    paramClassDefaults = TransferMechanism.paramClassDefaults.copy()

    standard_output_states = TransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:ENERGY}, {NAME:ENTROPY}])

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 matrix=FULL_CONNECTIVITY_MATRIX,
                 function=Linear,
                 initial_value=None,
                 decay:is_numeric_or_none=None,
                 noise:is_numeric_or_none=0.0,
                 time_constant:is_numeric_or_none=1.0,
                 range=None,
                 output_states:tc.optional(tc.any(list, dict))=[RESULT],
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Instantiate RecurrentTransferMechanism
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  matrix=matrix,
                                                  decay=decay,
                                                  output_states=output_states,
                                                  params=params,
                                                  noise=noise)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        super().__init__(default_input_value=default_input_value,
                         size=size,
                         input_states=input_states,
                         function=function,
                         initial_value=initial_value,
                         noise=noise,
                         time_constant=time_constant,
                         range=range,
                         output_states=output_states,
                         time_scale=time_scale,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of matrix and decay.
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

            elif isinstance(matrix_param, np.matrix):
                matrix = np.array(matrix_param)

            else:
                matrix = matrix_param

            rows = np.array(matrix).shape[0]
            cols = np.array(matrix).shape[1]

            # Shape of matrix must be square
            if rows != cols:
                if (matrix_param, MappingProjection):
                    # if __name__ == '__main__':
                    err_msg = ("{} param of {} must be square to be used as recurrent projection for {}".
                               format(MATRIX, matrix_param.name, self.name))
                else:
                    err_msg = "{} param for must be square".format(MATRIX, self.name)
                raise RecurrentTransferError(err_msg)

            # Size of matrix must equal length of variable:
            if rows != size:
                if (matrix_param, MappingProjection):
                    # if __name__ == '__main__':
                    err_msg = ("Number of rows in {} param for {} ({}) must be same as the size of the variable for {} "
                               "(whose size is {}, variable is {})".
                               format(MATRIX, self.name, rows, self.name, self.size, self.variable))
                else:
                    err_msg = ("Size of {} param for {} ({}) must same as its variable ({})".
                               format(MATRIX, self.name, rows, size))
                raise RecurrentTransferError(err_msg)


        if DECAY in target_set and target_set[DECAY] is not None:

            decay = target_set[DECAY]
            if not (0.0 <= decay and decay <= 1.0):
                raise RecurrentTransferError("{} argument for {} ({}) must be from 0.0 to 1.0".
                                             format(DECAY, self.name, decay))

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate recurrent_projection, matrix, and the functions for the ENERGY and ENTROPY outputStates
        """

        super()._instantiate_attributes_after_function(context=context)

        if isinstance(self.matrix, MappingProjection):
            self.recurrent_projection = self.matrix

        else:
            self.recurrent_projection = _instantiate_recurrent_projection(self, self.matrix, context=context)

        self._matrix = self.recurrent_projection.matrix

        if ENERGY in self.output_states.names:
            energy = Stability(self.variable[0],
                               metric=ENERGY,
                               transfer_fct=self.function,
                               matrix=self.recurrent_projection._parameter_states[MATRIX])
            self.output_states[ENERGY]._calculate = energy.function

        if ENTROPY in self.output_states.names:
            if self.function_object.bounds == (0,1) or range == (0,1):
                entropy = Stability(self.variable[0],
                                    metric=ENTROPY,
                                    transfer_fct=self.function,
                                    matrix=self.recurrent_projection._parameter_states[MATRIX])
                self.output_states[ENTROPY]._calculate = entropy.function
            else:
                del self.output_states[ENTROPY]

    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale = TimeScale.TRIAL,
                 context=None):
        """Implement decay
        """

        if INITIALIZING in context:
            self.previous_input = self.variable

        if self.decay is not None and self.decay != 1.0:
            self.previous_input *= self.decay

        return super()._execute(variable=variable,
                                runtime_params=runtime_params,
                                clock=CentralClock,
                                time_scale=time_scale,
                                context=context)


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
@tc.typecheck
def _instantiate_recurrent_projection(mech:Mechanism_Base,
                                      matrix:is_matrix=FULL_CONNECTIVITY_MATRIX,
                                      context=None):
    """Instantiate a MappingProjection from mech to itself

    """

    if isinstance(matrix, str):
        size = len(mech.variable[0])
        matrix = get_matrix(matrix, size, size)

    return MappingProjection(sender=mech,
                             receiver=mech,
                             matrix=matrix,
                             name = mech.name + ' recurrent projection')

