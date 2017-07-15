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

# ****************************************  LCA *************************************************

"""

Overview
--------

An LCA is a subclass of `RecurrentTransferMechanism` that implements a single-layered leaky competitive accumulator
network, in which each element is connected to every other element with mutually inhibitory weights.  All of the
inhibitory weights have the same value, specified by its `inhibition <LCA.inhibition>` parameter.  In the case that
it has two elements, the value of its `inhibition <LCA.inhibition>` parameter is equal to its `decay
<RecurrentTransferMechanism.decay>` parameter, and the two are of sufficient magnitude, it implements a close
approximation of a `DDM` Mechanism
(see `Usher & McClelland, 2001; <http://psycnet.apa.org/?&fa=main.doiLanding&doi=10.1037/0033-295X.108.3.550>`_ and
`Bogacz et al (2006) <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_).

.. _Recurrent_Transfer_Creation:

Creating an LCA
---------------

An LCA can be created directly by calling its constructor, or using the `mechanism() <Mechanism.mechanism>` function
and specifying LCA as its **mech_spec** argument.  The set of mutually inhibitory connections are implemented as a
recurrent `MappingProjection` with a `matrix <LCA.matrix>` of uniform negative weights specified by
the **inhibition** argument of the LCA's constructor.  The default format of its `variable <LCA.variable>`, and default
values of its `inhibition <LCA.inhibition>`, `decay <RecurrentTransferMechanism.decay>` and
`noise <TransferMechanism.noise>` parameters implement an approximation of a `DDM`.

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

from PsyNeuLink.Components.Functions.Function import Logistic, max_vs_next, max_vs_avg
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import *
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.States.OutputState import StandardOutputStates, PRIMARY_OUTPUT_STATE
import warnings


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
        default_input_value=None,          \
        size=None,                         \
        function=Logistic,                 \
        initial_value=None,                \
        decay=1.0,                         \
        inhibition=1.0,                    \
        noise=0.0,                         \
        time_constant=1.0,                 \
        range=(float:min, float:max),      \
        time_scale=TimeScale.TIME_STEP,    \
        params=None,                       \
        name=None,                         \
        prefs=None)

    Implements LCA subclass of `RecurrentTransferMechanism`.

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

    default_input_value : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    inhibition : number : default 1.0
        specifies the magnitude of the (uniform) negative weights used for the
        `matrix <LCA.matrix>` parameter of the `recurrent_projection <LCA.recurrent_projection>`.

    decay : number : default 1.0
        specifies the amount by which to decrement its `previous_input <TransferMechanism.previous_input>`
        in each execution of the Mechanism.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `time_constant <TransferMechanism.time_constant>` is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the Mechanism is executed with `time_scale`
        set to `TimeScale.TIME_STEP`

        `result = (time_constant * current input) + (1-time_constant * result on previous time_step)`

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <TransferMechanism.range>` that it exceeds.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the Mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.
        This must be set to `TimeScale.TIME_STEP` for the `time_constant <TransferMechanism.time_constant>`
        parameter to have an effect.

    name : str : default TransferMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    context : str : default ''componentType+INITIALIZNG''
           string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to the Mechanism's `function <LCA.function>`.

    function : Function
        the function used to transform the input.

    matrix : 2d np.array
        the `matrix <MappingProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism,
        with a uniform set of negative weights, the magnitude of which are determined by the
        `inhibition <LCA.inhibition>` attribute.

    recurrent_projection : MappingProjection
        a `MappingProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        back to it `primary inputState <Mechanism_InputStates>`.

    inhibition : number : default 1.0
        determines the magnitude of the (uniform) negative weights for the `matrix <LCA.matrix>` parameter
        of the `recurrent_projection <LCA.recurrent_projection>`.

    decay : float : default 1.0
        determines the amount by which to multiply the `previous_input <TransferMechanism.previous_input>` value
        in each execution of the Mechanism (acts, in effect like the weight on a self-connection).

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
        the value of the input on the previous execution of the Mechanism, including the value of
        `recurrent_projection`.

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`; same value as fist item of
        `output_values <TransferMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of `function <LCA.function>`;  also assigned to :keyword:`value` of the TRANSFER_RESULT
            OutputState and the first item of :keyword:`output_values`.
    COMMENT

    outputStates : Dict[str, OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:
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

    time_scale :  TimeScale
        specifies whether the Mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default TransferMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Returns
    -------
    instance of LCA : LCA

    """
    componentType = LCA

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    variableClassDefault = [[0]]

    # paramClassDefaults[OUTPUT_STATES].append({NAME:MAX_VS_NEXT})
    # paramClassDefaults[OUTPUT_STATES].append({NAME:MAX_VS_AVG})
    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:MAX_VS_NEXT,
                                    CALCULATE:max_vs_next},
                                   {NAME:MAX_VS_AVG,
                                    CALCULATE:max_vs_avg}])

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 size:tc.optional(tc.any(int, list, np.array))=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 matrix=None,
                 auto:is_numeric_or_none=None,  # not used: only here to avoid bugs
                 cross:is_numeric_or_none=None,
                 function=Logistic,
                 initial_value=None,
                 decay:tc.optional(tc.any(int, float))=1.0,
                 inhibition:tc.optional(tc.any(int, float))=1.0,
                 noise:is_numeric_or_none=0.0,
                 time_constant:is_numeric_or_none=1.0,
                 range=None,
                 output_states:tc.optional(tc.any(list, dict))=[RESULT],
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Instantiate LCA
        """

        # this may be problematic
        # IMPLEMENTATION NOTE: parts of this region may be redundant with code in ProcessingMechanism.__init__()
        # region Fill in and infer default_input_value and size if they aren't specified in args
        if default_input_value is None and size is None:
            default_input_value = self.variableClassDefault
            size = [1]

        # 6/23/17: This conversion is safe but likely redundant. If, at some point in development, size and
        # default_input_value are no longer 2D or 1D arrays, this conversion should still be safe, but wasteful.
        # region Convert default_input_value (if given) to a 2D array, and size (if given) to a 1D integer array

        try:
            if default_input_value is not None:
                default_input_value = np.atleast_2d(default_input_value)
                if len(np.shape(default_input_value)) > 2:  # number of dimensions of default_input_value > 2
                    warnings.warn("default_input_value had more than two dimensions (had {} dimensions) "
                                  "so only the first element of its second-highest-numbered axis will be"
                                  " used".format(len(np.shape(default_input_value))))
                    while len(np.shape(default_input_value)) > 2:  # reduce the dimensions of default_input_value
                        default_input_value = default_input_value[0]
        except:
            raise TransferError("Failed to convert default_input_value (of type {})"
                                " to a 2D array".format(type(default_input_value)))

        try:
            if size is not None:
                size = np.atleast_1d(size)
                if len(np.shape(size)) > 1:  # number of dimensions of size > 1
                    warnings.warn("size had more than one dimension (size had {} dimensions), so only the first "
                                  "element of its highest-numbered axis will be used".format(len(np.shape(size))))
                    while len(np.shape(size)) > 1:  # reduce the dimensions of size
                        size = size[0]
        except:
            raise TransferError("Failed to convert size (of type {}) to a 1D array.".format(type(size)))

        try:
            if size is not None:
                map(lambda x: int(x), size)  # convert all elements of size to int
        except:
            raise TransferError("Failed to convert an element in size to an integer.")
        # endregion

        # region If default_input_value is None, make it a 2D array of zeros each with length=size[i]
        # IMPLEMENTATION NOTE: perhaps add setting to enable user to change
        # default_input_value's default value, which is an array of zeros at the moment
        if default_input_value is None and size is not None:
            try:
                default_input_value = []
                for s in size:
                    default_input_value.append(np.zeros(s))
                default_input_value = np.array(default_input_value)
            except:
                raise TransferError("default_input_value was not specified, but PsyNeuLink was unable to "
                                    "infer default_input_value from the size argument, {}. size should be"
                                    " an integer or an array or list of integers. Either size or "
                                    "default_input_value must be specified.".format(size))
        # endregion

        # region If size is None, then make it a 1D array of scalars with size[i] = length(default_input_value[i])
        if size is None:
            size = []
            try:
                for input_vector in default_input_value:
                    size.append(len(input_vector))
                size = np.array(size)
            except:
                raise TransferError("size was not specified, but PsyNeuLink was unable to infer size from "
                                    "the default_input_value argument, {}. default_input_value can be an array,"
                                    " list, a 2D array, a list of arrays, array of lists, etc. Either size or"
                                    " default_input_value must be specified.".format(default_input_value))
        # endregion

        # region If length(size) = 1 and default_input_value is not None, then expand size to length(default_input_value)
        if len(size) == 1 and len(default_input_value) > 1:
            new_size = np.empty(len(default_input_value))
            new_size.fill(size[0])
            size = new_size
        # endregion

        # IMPLEMENTATION NOTE: if default_input_value and size are both specified as arguments, they will be checked
        # against each other in Component.py, during _instantiate_defaults().
        # endregion

        if matrix is not None:
            warnings.warn("Matrix arg for LCA is not used; matrix was assigned using inhibition arg")
        matrix = np.full((size[0], size[0]), -inhibition) * get_matrix(HOLLOW_MATRIX,size[0],size[0])

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  inhibition=inhibition,
                                                  output_states=output_states,
                                                  params=params)

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)


        super().__init__(default_input_value=default_input_value,
                         size=size,
                         input_states=input_states,
                         matrix=matrix,
                         function=function,
                         initial_value=initial_value,
                         decay=decay,
                         noise=noise,
                         time_constant=time_constant,
                         range=range,
                         output_states=output_states,
                         time_scale=time_scale,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)
