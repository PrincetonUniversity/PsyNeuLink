# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND BETA ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ****************************************  LCA *************************************************

"""

Overview
--------

An LCA is a subclass of `RecurrentTransferMechanism` that implements a single-layered leaky competitive accumulator
network, in which each element is connected to every other element with mutually inhibitory weights.  All of the
inhibitory weights have the same value, specified by its `competition <LCA.competition>` parameter.  In the case that
it has two elements, the value of its `competition <LCA.competition>` parameter is equal to its `leak
<LCA.leak>` parameter, and the two are of sufficient magnitude, it implements a close
approximation of a `DDM` Mechanism
(see `Usher & McClelland, 2001; <http://psycnet.apa.org/?&fa=main.doiLanding&doi=10.1037/0033-295X.108.3.550>`_ and
`Bogacz et al (2006) <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_).

.. note::

    The LCA's recurrent projection matrix always has self_excitation on the diagonal and -competition off-diagonal.

    COMMENT:
    .. math::

        \\begin{bmatrix}
            excitation    &  - competition  &  - competition  &  - competition  \
            - competition &  excitation     &  - competition  &  - competition  \
            - competition &  - competition  &  excitation     &  - competition  \
            - competition &  - competition  &  - competition  &  excitation     \
        \\end{bmatrix}
    COMMENT

.. _Recurrent_Transfer_Creation:

Creating an LCA
---------------

An LCA can be created directly by calling its constructor.  The set of mutually inhibitory connections are implemented as a recurrent `MappingProjection`
with a `matrix <LCA.matrix>` in which the diagonal consists of uniform weights specified by **self_excitation** and the
off-diagonal consists of uniform *negative* weights specified by the **competition** argument of the LCA's constructor.

COMMENT:
.. math::

    \\begin{bmatrix}
        excitation    &  - competition  &  - competition  &  - competition  \
        - competition &  excitation     &  - competition  &  - competition  \
        - competition &  - competition  &  excitation     &  - competition  \
        - competition &  - competition  &  - competition  &  excitation     \
    \\end{bmatrix}
COMMENT

COMMENT:
The default format of its `variable <LCA.variable>`, and default values of its `inhibition
<LCA.inhibition>`, `decay <RecurrentTransferMechanism.decay>` and `noise <TransferMechanism.noise>` parameters
implement an approximation of a `DDM`.
COMMENT

.. _LCA_Structure:

Structure
---------

The distinguishing feature of an LCA is its `matrix <LCA.matrix>` of uniform negative weights.  It also has, in
addition to its `primary OutputState <OutputState_Primary>` (which contains the current value of the elements of the
LCA) and the OutputStates of a RecurrentTransferMechanism, it has two additional OutputStates: `MAX_VS_NEXT <LCA.LCA_OUTPUT.MAX_VS_NEXT>` and
`MAX_VS_AVG <MAX_VS_AVG>`.  Both are two element arrays that track the element of the LCA with the currently highest value relative
to the value of the others.  The two elements of the `MAX_VS_NEXT` OutputState contain, respectively, the index of the
LCA element with the greatest value, and the difference between its value and the next highest one;  `MAX_VS_AVG`
contains the index of the LCA element with the greatest value, and the difference between its value and the average
of all the others.  For an LCA with only two elements, `MAX_VS_NEXT` implements a close approximation of the
`threshold <DDM.threshold>` parameter of a `DDM`
(see `Usher & McClelland, 2001; <http://psycnet.apa.org/?&fa=main.doiLanding&doi=10.1037/0033-295X.108.3.550>`_ and
`Bogacz et al (2006) <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_).
For an LCA with more than two elements, `MAX_VS_NEXT` and
`MAX_VS_AVERAGE` implement threshold approximations with different properties
(see `McMillen & Holmes, 2006 <http://www.sciencedirect.com/science/article/pii/S0022249605000891>`_).

.. _LCA_Execution:

Execution
---------

The execution of an LCA is identical to that of `RecurrentTransferMechanism`.

.. _LCA_Class_Reference:

Class Reference
---------------


"""

import warnings
from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import LCAIntegrator, Logistic, max_vs_avg, max_vs_next
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.globals.keywords import TIME_STEP_SIZE, BETA, CALCULATE, ENERGY, ENTROPY, INITIALIZER, INITIALIZING, LCA, MEAN, MEDIAN, NAME, NOISE, RATE, RESULT, STANDARD_DEVIATION, VARIANCE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.scheduling.timescale import TimeScale

__all__ = [
    'LCA', 'LCA_OUTPUT', 'LCAError', 'MAX_VS_AVG', 'MAX_VS_NEXT',
]


class LCAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

MAX_VS_NEXT = 'max_vs_next'
MAX_VS_AVG = 'max_vs_avg'

# This is a convenience class that provides list of standard_output_state names in IDE
class LCA_OUTPUT():
        """
            .. _LCA_Standard_OutputStates:

            `Standard OutputStates <OutputState_Standard>` for `LCA`:

            .. _LCA_RESULT

            *RESULT* : 1d np.array
                result of the `function <LCA.function>` calculation

            .. _LCA_MEAN

            *MEAN* : float
                the mean of the result

            .. _LCA_VARIANCE

            *VARIANCE* : float
                the variance of the result

            .. _LCA_ENERGY

            *ENERGY* : float
                the energy of the result, which is calculated using the `Stability
                Function <Function.Stability.function>` with the ``ENERGY`` metric

            .. _LCA_ENTROPY

            *ENTROPY* : float
                the entropy of the result, which is calculated using the `Stability
                Function <Function.Stability.function>` with ``ENTROPY`` metric (Note:
                this is only present if the Mechanism's `function` is bounded between
                0 and 1)

            .. _LCA_MAX_VS_NEXT

            *MAX_VS_NEXT* : 1d np.array
                a two-element Numpy array containing the index of the element of
                `RESULT <LCA_OUTPUT.RESULT>` with the highest value (element 1) and the difference
                between that and the next highest one in `TRANSFER_RESULT` (element 2)

            .. _LCA_MAX_VS_AVG

            *MAX_VS_AVG* : 1d np.array
                a two-element Numpy array containing the index of the element of
                `RESULT` with the highest value (element 1) and the difference
                between that and the average of the value of all of `RESULT`'s
                other elements
        """
        RESULT=RESULT
        MEAN=MEAN
        MEDIAN=MEDIAN
        STANDARD_DEVIATION=STANDARD_DEVIATION
        VARIANCE=VARIANCE
        ENERGY=ENERGY
        ENTROPY=ENTROPY
        MAX_VS_NEXT=MAX_VS_NEXT
        MAX_VS_AVG=MAX_VS_AVG
# THIS WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
# for item in [item[NAME] for item in DDM_standard_output_states]:
#     setattr(DDM_OUTPUT.__class__, item, item)


    # THIS WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
    # for item in [item[NAME] for item in DDM_standard_output_states]:
    #     setattr(DDM_OUTPUT.__class__, item, item)


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class LCA(RecurrentTransferMechanism):
    """
    LCA(                                   \
        default_variable=None,             \
        size=None,                         \
        function=Logistic,                 \
        initial_value=None,                \
        leak=0.5,                          \
        competition=1.0,                   \
        self_excitation=0.0,               \
        noise=0.0,                         \
        clip=(float:min, float:max),       \
        params=None,                       \
        name=None,                         \
        prefs=None)                        \

    Subclass of `RecurrentTransferMechanism` that implements a Leaky Competitive Accumulator.

    COMMENT:
        Description
        -----------
            LCA is a Subtype of the RecurrentTransferMechanism Subtype of the TransferMechanism
            Subtype of the ProcessingMechanisms Type of the Mechanism Category of the Component class.
            It implements a RecurrentTransferMechanism with a set of uniform recurrent inhibitory weights.
            In all other respects, it is identical to a RecurrentTransferMechanism.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
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

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `beta <TransferMechanism.beta>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    leak : value : default 0.5
        sets the `rate <LCAIntegrator.rate>` on the `LCAIntegrator function <LCAIntegrator>`, which scales the
        contribution of the `LCAIntegrator's <LCAIntegrator>` `previous_value <LCAIntegrator.previous_value>` to the
        accumulation of the `LCAIntegrator's value <LCAIntegrator.value>` (:math:`x_{i}`) on each time step. See
        `LCAIntegrator <LCAIntegrator>` for more details on what the `LCAIntegrator function  <LCAIntegrator>` computes.

    competition : value : default 1.0
        sets the magnitude of the off-diagonal terms in the LCA's recurrent projection, thereby scaling the
        contributions of the competing unit (all :math:`f(x)_{j}` where :math:`j \\neq i`) to the accumulation of the
        `LCAIntegrator's value <LCAIntegrator.value>` (:math:`x_{i}`) on each time step. See
        `LCAIntegrator <LCAIntegrator>` for more details on what the `LCAIntegrator function  <LCAIntegrator>` computes.

    self_excitation : value : default 0.0
        sets the diagonal terms in the LCA's recurrent projection, thereby scaling the contributions of each unit's own
        recurrent value (:math:`f(x)_{i}`) to the accumulation of the `LCAIntegrator's value <LCAIntegrator.value>`
        (:math:`x_{i}`) on each time step. See `LCAIntegrator <LCAIntegrator>` for more details on what the
        `LCAIntegrator function  <LCAIntegrator>` computes.

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    clip : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <TransferMechanism.clip>` that it exceeds.

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <LCA Mechanism.name>`
        specifies the name of the LCA Mechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LCA Mechanism; see `prefs <LCA Mechanism.prefs>` for details.

    context : str : default ''componentType+INITIALIZNG''
           string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to the Mechanism's `function <LCA.function>`.

    function : Function
        the function used to transform the input.

    matrix : 2d np.array
        the `matrix <MappingProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism, the
        `self_excitation <LCA.self_excitation>` attribute sets the values on the diagonal, and the
        `competition <LCA.competition>` attribute sets the magnitude of the negative off-diagonal values.

    leak : value : default 0.5
        sets the `rate <LCAIntegrator.rate>` on the `LCAIntegrator function <LCAIntegrator>`, which scales the
        contribution of the `LCAIntegrator's <LCAIntegrator>` `previous_value <LCAIntegrator.previous_value>` to the
        accumulation of the `LCAIntegrator's value <LCAIntegrator.value>` (:math:`x_{i}`) on each time step. See
        `LCAIntegrator <LCAIntegrator>` for more details on what the `LCAIntegrator function  <LCAIntegrator>` computes.

    competition : value : default 1.0
        sets the magnitude of the off-diagonal terms in the LCA's recurrent projection, thereby scaling the
        contributions of the competing unit (all :math:`f(x)_{j}` where :math:`j \\neq i`) to the accumulation of the
        `LCAIntegrator's value <LCAIntegrator.value>` (:math:`x_{i}`) on each time step. See
        `LCAIntegrator <LCAIntegrator>` for more details on what the `LCAIntegrator function  <LCAIntegrator>` computes.

    self_excitation : value : default 0.0
        sets the diagonal terms in the LCA's recurrent projection, thereby scaling the contributions of each unit's own
        recurrent value (:math:`f(x)_{i}`) to the accumulation of the `LCAIntegrator's value <LCAIntegrator.value>`
        (:math:`x_{i}`) on each time step. See `LCAIntegrator <LCAIntegrator>` for more details on what the
        `LCAIntegrator function  <LCAIntegrator>` computes.

    recurrent_projection : MappingProjection
        a `MappingProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        back to it `primary inputState <Mechanism_InputStates>`.


    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `beta <TransferMechanism.beta>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <LCA.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    clip : Tuple[float, float]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `clip <TransferMechanism.clip>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `clip <TransferMechanism.clip>` is set by default to (0,1).

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`; same value as fist item of
        `output_values <TransferMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of `function <LCA.function>`;  also assigned to :keyword:`value` of the TRANSFER_RESULT
            OutputState and the first item of :keyword:`output_values`.
    COMMENT

    output_states : ContentAddressableList[OutputState]
        contains the following `OutputStates <OutputState>`:
        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <TransferMechanism.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;
        * `ENERGY`, the :keyword:`value` of which is the energy of the result,
          calculated using the `Stability` Function with the ENERGY metric;
        * `ENTROPY`, the :keyword:`value` of which is the entropy of the result,
          calculated using the `Stability` Function with the ENTROPY metric;
          note:  this is only present if the Mechanism's :keyword:`function` is bounded between 0 and 1
          (e.g., the `Logistic` function);
        * `MAX_VS_NEXT`, the :keyword:`value` of which is a two element array: the first is the
          index of the element of RESULT with the highest value, and the second the difference between that
          and the next highest one in RESULT;
        * `MAX_VS_AVG`, the :keyword:`value` of which is a two element array: the first is the
          index of the element of RESULT with the highest value, and the second the difference between that
          and the average of the value of all its other elements;

    output_values : List[array(float64), float, float]
        a list with the following items:
        * **result** of the `function <LCA.function>` calculation (value of TRANSFER_RESULT OutputState);
        * **mean** of the result (:keyword:`value` of TRANSFER_MEAN OutputState)
        * **variance** of the result (:keyword:`value` of TRANSFER_VARIANCE OutputState);
        * **energy** of the result (:keyword:`value` of ENERGY OutputState);
        * **entropy** of the result (if the ENTROPY OutputState is present);
        * **max_vs_next** of the result (:keyword:`value` of MAX_VS_NEXT OutputState);
        * **max_vs_avg** of the result (:keyword:`value` of MAX_VS_AVG OutputState).

    name : str
        the name of the LCA Mechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LCA Mechanism; if it is not specified in the **prefs** argument of the 
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet 
        <LINK>` for details).

    Returns
    -------
    instance of LCA : LCA

    """
    componentType = LCA

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        # RATE: None,
        BETA: None
    })
    class ClassDefaults(RecurrentTransferMechanism.ClassDefaults):
        variable = [[0]]

    # paramClassDefaults[OUTPUT_STATES].append({NAME:MAX_VS_NEXT})
    # paramClassDefaults[OUTPUT_STATES].append({NAME:MAX_VS_AVG})
    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:MAX_VS_NEXT,
                                    CALCULATE:max_vs_next},
                                   {NAME:MAX_VS_AVG,
                                    CALCULATE:max_vs_avg}])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size:tc.optional(tc.any(int, list, np.array))=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 matrix=None,
                 function=Logistic,
                 initial_value=None,
                 leak=0.5,
                 competition=1.0,
                 self_excitation=0.0,
                 noise=0.0,
                 integrator_mode=True,
                 time_step_size=0.1,
                 clip=None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULT,
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Instantiate LCA
        """

        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None or output_states is RESULT:
            output_states = [RESULT]

        if matrix is not None:
            warnings.warn("Matrix arg for LCA is not used; matrix was assigned using self_excitation and competition "
                          "args")
        # matrix = np.full((size[0], size[0]), -inhibition) * get_matrix(HOLLOW_MATRIX,size[0],size[0])

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  leak=leak,
                                                  self_excitation=self_excitation,
                                                  competition=competition,
                                                  integrator_mode=integrator_mode,
                                                  time_step_size=time_step_size,
                                                  output_states=output_states,
                                                  params=params)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)


        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         auto=self_excitation,
                         hetero=-competition,
                         function=function,
                         initial_value=initial_value,
                         noise=noise,
                         clip=clip,
                         output_states=output_states,
                         time_scale=time_scale,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=None,
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
            + BETA (float)
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
        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        integrator_mode = self.integrator_mode

        #region ASSIGN PARAMETER VALUES

        beta = self.beta
        clip = self.clip
        noise = self.noise
        time_step_size = self.time_step_size

        #endregion
        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        # Update according to time-scale of integration
        if integrator_mode:
        # if time_scale is TimeScale.TIME_STEP:

            if not self.integrator_function:

                self.integrator_function = LCAIntegrator(
                                            variable,
                                            initializer=self.initial_value,
                                            noise=self.noise,
                                            time_step_size=self.time_step_size,
                                            rate=self.leak,
                                            owner=self)

            current_input = self.integrator_function.execute(variable,
                                                        # Should we handle runtime params?
                                                              params={INITIALIZER: self.initial_value,
                                                                      NOISE: self.noise,
                                                                      RATE: self.leak,
                                                                      TIME_STEP_SIZE: self.time_step_size},
                                                              context=context

                                                             )
        else:
        # elif time_scale is TimeScale.TRIAL:
            noise = self._try_execute_param(self.noise, variable)
            # formerly: current_input = self.input_state.value + noise
            # (MODIFIED 7/13/17 CW) this if/else below is hacky: just allows a nicer error message
            # when the input is given as a string.
            if (np.array(noise) != 0).any():
                current_input = variable[0] + noise
            else:

                current_input = variable[0]

        # Apply TransferMechanism function
        output_vector = self.function(variable=current_input, params=runtime_params)

        if clip is not None:
            minCapIndices = np.where(output_vector < clip[0])
            maxCapIndices = np.where(output_vector > clip[1])
            output_vector[minCapIndices] = np.min(clip)
            output_vector[maxCapIndices] = np.max(clip)

        return output_vector
    # @property
    # def inhibition(self):
    #     return self.hetero
    #
    # @inhibition.setter
    # def inhibition(self, setting):
    #     self.hetero = setting
