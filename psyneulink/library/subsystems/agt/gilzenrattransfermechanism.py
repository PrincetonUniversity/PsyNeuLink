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

# ****************************************  GilzenratTransferMechanism *************************************************

import numbers

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import Hebbian, Linear, is_function_type, LCAIntegrator
from psyneulink.components.mechanisms.adaptive.learning.learningmechanism import LearningMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.globals.keywords import INITIALIZER, NOISE, RATE, ENERGY, ENTROPY, FULL_CONNECTIVITY_MATRIX, INITIALIZING, MEAN, MEDIAN, NAME, RECURRENT_TRANSFER_MECHANISM, RESULT, STANDARD_DEVIATION, VARIANCE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.scheduling.timescale import CentralClock, TimeScale

class GilzenratTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

DECAY = 'decay'

# This is a convenience class that provides list of standard_output_state names in IDE
class RECURRENT_OUTPUT():

    """
        .. _GilzenratTransferMechanism_Standard_OutputStates:

        `Standard OutputStates <OutputState_Standard>` for
        `GilzenratTransferMechanism`

        .. TRANSFER_RESULT:

        *RESULT* : 1d np.array
            the result of the `function <GilzenratTransferMechanism.function>`
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
class GilzenratTransferMechanism(RecurrentTransferMechanism):
    """
    GilzenratTransferMechanism(        \
    default_variable=None,             \
    size=None,                         \
    function=Linear,                   \
    matrix=FULL_CONNECTIVITY_MATRIX,   \
    auto=None,                         \
    hetero=None,                       \
    initial_value=None,                \
    decay=None,                        \
    noise=0.0,                         \
    time_constant=1.0,                 \
    range=(float:min, float:max),      \
    learning_rate=None,                \
    learning_function=Hebbian,         \
    params=None,                       \
    name=None,                         \
    prefs=None)

    Subclass of `TransferMechanism` that implements a single-layer auto-recurrent network.

    COMMENT:
        Description
        -----------
            GilzenratTransferMechanism is a Subtype of the TransferMechanism Subtype of the ProcessingMechanisms Type
            of the Mechanism Category of the Component class.
            It implements a TransferMechanism with a recurrent projection (default matrix: FULL_CONNECTIVITY_MATRIX).
            In all other respects, it is identical to a TransferMechanism.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <GilzenratTransferMechanism.variable>` for
        `function <GilzenratTransferMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <Recurrent_Transfer_Structure>`,
        or an AutoAssociativeProjection to use. If **auto** or **hetero** arguments are specified, the **matrix**
        argument will be ignored in favor of those arguments.

    auto : number, 1D array, or None : default None
        specifies matrix as a diagonal matrix with diagonal entries equal to **auto**, if **auto** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. If the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **auto** can be specified as a 1D array with length equal
        to the size of the Mechanism, if a non-uniform diagonal is desired. Can be modified by control.

    hetero : number, 2D array, or None : default None
        specifies matrix as a hollow matrix with all non-diagonal entries equal to **hetero**, if **hetero** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. If the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **hetero** can be specified as a 2D array with dimensions
        equal to the matrix dimensions, if a non-uniform diagonal is desired. Can be modified by control.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `time_constant <GilzenratTransferMechanism.time_constant>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    decay : number : default 1.0
        specifies the amount by which to decrement its `previous_input <GilzenratTransferMechanism.previous_input>`
        each time it is executed.

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <GilzenratTransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when `integrator_mode
        <GilzenratTransferMechanism.integrator_mode>` is set to True::

         result = (time_constant * current input) +
         (1-time_constant * result on previous time_step)

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <GilzenratTransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <GilzenratTransferMechanism.range>` that it exceeds.

    enable_learning : boolean : default False
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <GilzenratTransferMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <GilzenratTransferMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <GilzenratTransferMechanism.learning_rate>` for details).

    learning_function : function : default Hebbian
        specifies the function for the LearningMechanism if `learning has been specified
        <Recurrent_Transfer_Learning>` for the GilzenratTransferMechanism.  It can be any function so long as it
        takes a list or 1d array of numeric values as its `variable <Function_Base.variable>` and returns a sqaure
        matrix of numeric values with the same dimensions as the length of the input.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its `function <GilzenratTransferMechanism.function>`, and/or a custom function and its
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str : default see `name <GilzenratTransferMechanism.name>`
        specifies the name of the GilzenratTransferMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the GilzenratTransferMechanism; see `prefs <GilzenratTransferMechanism.prefs>` 
        for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <GilzenratTransferMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        back to its `primary inputState <Mechanism_InputStates>`.

    decay : float : default 1.0
        determines the amount by which to multiply the `previous_input <GilzenratTransferMechanism.previous_input>`
        value each time it is executed.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input (only relevant if `time_constant
        <GilzenratTransferMechanism.time_constant>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <GilzenratTransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float
        the time constant for exponential time averaging of input when `integrator_mode
        <GilzenratTransferMechanism.integrator_mode>` is set to True::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Tuple[float, float]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <GilzenratTransferMechanism.range>` it exceeds.  If
        `function <GilzenratTransferMechanism.function>`
        is `Logistic`, `range <GilzenratTransferMechanism.range>` is set by default to (0,1).

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    learning_enabled : bool : default False
        indicates whether learning has been enabled for the GilzenratTransferMechanism.  It is set to `True` if
        `learning is specified <Recurrent_Transfer_Learning>` at the time of construction (i.e., if the
        **enable_learning** argument of the Mechanism's constructor is assigned `True`, or when it is configured for
        learning using the `configure_learning <GilzenratTransferMechanism.configure_learning>` method.  Once learning
        has been configured, `learning_enabled <RecurrentMechahinsm.learning_enabled>` can be toggled at any time to
        enable or disable learning; however, if the Mechanism has not been configured for learning, an attempt to
        set `learning_enabled <RecurrentMechahinsm.learning_enabled>` to `True` elicits a warning and is then
        ignored.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        specifies the learning rate used by the `learning_function <GilzenratTransferMechanism.learning_function>`
        of the `learning_mechanism <GilzenratTransferMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignement).

    learning_function : function : default Hebbian
        the function used by the `learning_mechanism <GilzenratTransferMechanism.learning_mechanism>` to train the
        `recurrent_projection <GilzenratTransferMechanism.recurrent_projection>` if `learning is specified
        <Recurrent_Transfer_Learning>`.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <Recurrent_Transfer_Learning>`, and used to train the
        `recurrent_projection <GilzenratTransferMechanism.recurrent_projection>`.

    value : 2d np.array [array(float64)]
        result of executing `function <GilzenratTransferMechanism.function>`; same value as first item of
        `output_values <GilzenratTransferMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    output_states : Dict[str, OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:

        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <GilzenratTransferMechanism.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;
        * `ENERGY`, the :keyword:`value` of which is the energy of the result,
          calculated using the `Stability` Function with the ENERGY metric;
        * `ENTROPY`, the :keyword:`value` of which is the entropy of the result,
          calculated using the `Stability` Function with the ENTROPY metric;
          note:  this is only present if the Mechanism's :keyword:`function` is bounded between 0 and 1
          (e.g., the `Logistic` function).

    output_values : List[array(float64), float, float]
        a list with the following items:

        * **result** of the ``function`` calculation (value of TRANSFER_RESULT OutputState);
        * **mean** of the result (``value`` of TRANSFER_MEAN OutputState)
        * **variance** of the result (``value`` of TRANSFER_VARIANCE OutputState);
        * **energy** of the result (``value`` of ENERGY OutputState);
        * **entropy** of the result (if the ENTROPY OutputState is present).

    name : str
        the name of the GilzenratTransferMechanism; if it is not specified in the **name** argument of the constructor,
        a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the GilzenratTransferMechanism; if it is not specified in the **prefs** argument of the 
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet 
        <LINK>` for details).

    Returns
    -------
    instance of GilzenratTransferMechanism : GilzenratTransferMechanism

    """
    componentType = RECURRENT_TRANSFER_MECHANISM

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()
    standard_output_states = TransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:ENERGY}, {NAME:ENTROPY}])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=Linear,
                 matrix=FULL_CONNECTIVITY_MATRIX,
                 auto=None,
                 hetero=None,
                 initial_value=None,
                 decay: is_numeric_or_none=None,
                 noise=0.0,
                 time_step_size=0.02,
                 range=None,
                 input_states: tc.optional(tc.any(list, dict)) = None,
                 enable_learning:bool=False,
                 learning_rate: tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = Hebbian,
                 output_states: tc.optional(tc.any(list, dict))=None,
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Instantiate GilzenratTransferMechanism
        """
        if output_states is None:
            output_states = [RESULT]

        if isinstance(hetero, (list, np.matrix)):
            hetero = np.array(hetero)

        self._learning_enabled = enable_learning

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  initial_value=initial_value,
                                                  matrix=matrix,
                                                  decay=decay,
                                                  learning_rate=learning_rate,
                                                  learning_function=learning_function,
                                                  output_states=output_states,
                                                  time_step_size=time_step_size,
                                                  params=params,
                                                  noise=noise,
                                                  auto=auto,
                                                  hetero=hetero)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         initial_value=initial_value,
                         noise=noise,
                         range=range,
                         output_states=output_states,
                         time_scale=time_scale,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)


    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to output_values:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.MEAN].value
            variance of items in RESULT outputState[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + TIME_CONSTANT (float)
            + RANGE ([float, float])
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.output_states list:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # FIX: ??CALL check_args()??

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.instance_defaults.variable to initialize state of input

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        time_scale = self.time_scale

        #region ASSIGN PARAMETER VALUES

        time_constant = self.time_constant
        range = self.range
        noise = self.noise

        #endregion

        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        if not self.integrator_function:

            self.integrator_function = LCAIntegrator(
                                        variable,
                                        initializer=self.initial_value,
                                        noise=self.noise,
                                        rate=-1.0,
                                        time_step_size=self.time_step_size,
                                        owner=self)

        current_input=self.integrator_function.execute(variable,
                                                    # Should we handle runtime params?
                                                          params={INITIALIZER: self.initial_value,
                                                                  NOISE: self.noise,
                                                                  RATE: -1.0},
                                                          context=context

                                                         )

        # Apply TransferMechanism function
        output_vector = self.function(variable=current_input,
                                      params=runtime_params)
        # # MODIFIED  OLD:
        # if list(range):
        # MODIFIED  NEW:
        if range is not None:
        # MODIFIED  END
            minCapIndices = np.where(output_vector < range[0])
            maxCapIndices = np.where(output_vector > range[1])
            output_vector[minCapIndices] = np.min(range)
            output_vector[maxCapIndices] = np.max(range)

        return output_vector
        #endregion

        # return super()._execute(variable=variable,
        #                         runtime_params=runtime_params,
        #                         clock=CentralClock,
        #                         time_scale=time_scale,
        #                         context=context)


