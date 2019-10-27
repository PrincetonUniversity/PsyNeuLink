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

# ****************************************  LCAMechanism *************************************************

"""

Overview
--------

An LCAMechanism is a subclass of `RecurrentTransferMechanism` that implements a single-layered `leaky competitng
accumulator (LCA) <https://www.ncbi.nlm.nih.gov/pubmed/11488378>`_  network, in which each element is connected to
every other element with mutually inhibitory weights. The LCAMechanism's recurrent projection matrix *always* consists
of `self_excitation <LCAMechanism.self_excitation>` on the diagonal and -`competition <LCAMechanism.competition>`
off-diagonal.

    COMMENT:
    .. math::

        \\begin{bmatrix}
            excitation    &  - competition  &  - competition  &  - competition  \
            - competition &  excitation     &  - competition  &  - competition  \
            - competition &  - competition  &  excitation     &  - competition  \
            - competition &  - competition  &  - competition  &  excitation     \
        \\end{bmatrix}
    COMMENT

When all of the following conditions are true:

- The `LCAMechanism` mechanism has two elements
- The value of its `competition <LCAMechanism.competition>` parameter is equal to its `leak <LCAMechanism.leak>`
  parameter
- `Competition <LCAMechanism.competition>` and `leak <LCAMechanism.leak>` are of sufficient magnitude

then the `LCAMechanism` implements a close approximation of a `DDM` Mechanism (see `Usher & McClelland, 2001;
<http://psycnet.apa.org/?&fa=main.doiLanding&doi=10.1037/0033-295X.108.3.550>`_ and `Bogacz et al (2006)
<https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_).

.. _Recurrent_Transfer_Creation:

Creating an LCAMechanism
------------------------

An LCAMechanism can be created directly by calling its constructor.

The self-excitatory and mutually-inhibitory connections are implemented as a recurrent `MappingProjection`
with a `matrix <LCAMechanism.matrix>` in which the diagonal consists of uniform weights specified by
**self_excitation** and the off-diagonal consists of uniform weights specified by the *negative* of the
**competition** argument.

.. _LCAMechanism_Integrator_Mode

*Integration*
~~~~~~~~~~~~~

The **noise**, **leak**, **initial_value**, and **time_step_size** arguments are used to implement the
`LeakyCompetingIntegrator` as the LCAMechanism's `integrator_function <TransferMechanism.integrator_function>`.
This is only used used when `integrator_mode <Transfer_Integrator_Mode>` is True.  If `integrator_mode
<TransferMechanism.integrator_mode>` is False, the `LeakyCompetingIntegrator` function is skipped entirely,
and all related arguments (**noise**, **leak**, **initial_value**, and **time_step_size**) have no effect.

.. _LCAMechanism_Threshold:

*Thresholding*
~~~~~~~~~~~~~~

The **threshold** and **threshold_criterion** arguments specify the conditions under which execution of the
LCAMechanism terminates if `integrator_mode <Transfer_Integrator_Mode>` is True. These are convenience arguments,
that are used to specify its `termination_threshold <TransferMechanism.termination_threshold>`,
`termination_measure <TransferMechanism.termination_measure>`,  and `termination_comparison_op
<TransferMechanism.termination_comparison_op>` attributes; these can also be specified directly as arguments of the
LCAMechanism's constructor (see `Transfer_Execution_Termination` for additional details).

COMMENT:
The default format of its `variable <LCAMechanism.variable>`, and default values of its `inhibition
<LCAMechanism.inhibition>`, `decay <RecurrentTransferMechanism.decay>` and `noise <TransferMechanism.noise>` parameters
implement an approximation of a `DDM`.
COMMENT

.. _LCAMechanism_Structure:

Structure
---------

The key distinguishing features of an LCAMechanism are:

1. its `integrator_function <TransferMechanism.integrator_function>`, which implements the `LeakyCompetingIntegrator`
Function by default (in place of the `AdaptiveIntegrator` used as the default by other `TransferMechanisms
<TransferMechanism>`).

2. its `matrix <LCAMechanism.matrix>` consisting of `self_excitation <LCAMechanism.self_excitation>` and `competition
<LCAMechanism.competition>` off diagonal.

In addition to its `primary OutputPort <OutputPort_Primary>` (which contains the current `value <Mechanism_Base.value>`
of the LCAMechanism), it has available the `StandardOutputPorts of a RecurrentTransferMechanism
<RecurrentTransferMechanism_Standard_OutputPorts>`, it has two additional `Standard OutputPorts <OutputPort_Standard>`:

    - `MAX_VS_NEXT <LCAMechanism_MAX_VS_NEXT>`

    - `MAX_VS_AVG <LCAMechanism_MAX_VS_AVG>`

COMMENT:
The two elements of the `MAX_VS_NEXT <LCAMechanism_MAX_VS_NEXT>` OutputPort contain, respectively, the index of the
LCAMechanism element with the greatest value, and the difference between its value and the next highest one.
`MAX_VS_AVG <LCAMechanism_MAX_VS_AVG>` contains the index of the LCAMechanism element with the greatest value,
and the difference between its  value and the average of all the others.
COMMENT
The `value <OutputPort.value>` of the `MAX_VS_NEXT <LCAMechanism_MAX_VS_NEXT>` OutputPort contains the difference
between the two elements of the LCAMechanism’s `value <Mechanism_Base.value>` with the highest values, and the `value
<OutputPort.value>` of the `MAX_VS_AVG <LCAMechanism_MAX_VS_AVG>` OutputPort contains the difference between the
element with the highest value and the average of all the others.

For an LCAMechanism with *exactly* two elements, `MAX_VS_NEXT <LCAMechanism_MAX_VS_NEXT>` implements a close
approximation of the `threshold <DriftDiffusionIntegrator.threshold>` parameter of the `DriftDiffusionIntegrator`
Function used by a `DDM` (see `Usher & McClelland, 2001;
<http://psycnet.apa.org/?&fa=main.doiLanding&doi=10.1037/0033-295X.108.3.550>`_ and `Bogacz et al (2006)
<https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_). For an LCAMechanism with more than two elements, `MAX_VS_NEXT
<LCAMechanism_MAX_VS_NEXT>` and `MAX_VS_AVG <LCAMechanism_MAX_VS_AVG>` implement threshold approximations with
different properties (see `McMillen & Holmes, 2006
<http://www.sciencedirect.com/science/article/pii/S0022249605000891>`_).

.. _LCAMechanism_Execution:

Execution
---------

The execution of an LCAMechanism is identical to that of `RecurrentTransferMechanism`.

.. _LCAMechanism_Class_Reference:

Class Reference
---------------


"""

import warnings
import logging

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.selectionfunctions import max_vs_avg, max_vs_next, MAX_VS_NEXT, MAX_VS_AVG
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import LeakyCompetingIntegrator
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.processing.transfermechanism import _integrator_mode_setter
from psyneulink.core.components.ports.outputport import PRIMARY, StandardOutputPorts
from psyneulink.core.globals.keywords import \
    BETA, ENERGY, ENTROPY, FUNCTION, GREATER_THAN_OR_EQUAL, INITIALIZER, LCA_MECHANISM, NAME, NOISE, \
    OUTPUT_MEAN, OUTPUT_MEDIAN, OUTPUT_STD_DEV, OUTPUT_VARIANCE, RATE, RESULT, \
    TERMINATION_THRESHOLD, TERMINATION_MEASURE, TERMINATION_COMPARISION_OP, THRESHOLD, TIME_STEP_SIZE, VALUE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferMechanism

__all__ = ['LCAMechanism', 'LCAMechanism_OUTPUT', 'LCAError']


logger = logging.getLogger(__name__)


class LCAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# This is a convenience class that provides list of standard_output_port names in IDE
class LCAMechanism_OUTPUT():
        """
            .. _LCAMechanism_Standard_OutputPorts:

            An LCAMechanism has the following `Standard OutputPorts <OutputPort_Standard>` in addition to those of a
            `RecurrentTransferMechanism <RecurrentTransferMechanism_Standard_OutputPorts>`:

            COMMENT:
            .. _LCAMechanism_RESULT:

            *RESULT* : 1d np.array
                the LCAMechanism's `value <Mechanism_Base.value>`.

            .. _LCAMechanism_MEAN:

            *OUTPUT_MEAN* : float
                the mean of the elements in the LCAMechanism's `value <Mechanism_Base.value>`;.

            .. _LCAMechanism_VARIANCE:

            *OUTPUT_VARIANCE* : float
                the variance of the elements in the LCAMechanism's `value <Mechanism_Base.value>`;.

            .. _LCAMechanism_ENERGY:

            *ENERGY* : float
                the energy of the elements in the LCAMechanism's `value <Mechanism_Base.value>`,
                calculated using the `Stability` Function using the `ENERGY` metric.

            .. _LCAMechanism_ENTROPY:

            *ENTROPY* : float
                the entropy of the elements in the LCAMechanism's `value <Mechanism_Base.value>`,
                calculated using the `Stability` Function using the `ENTROPY <CROSS_ENTROPY>` metric.
            COMMENT

            .. _LCAMechanism_MAX_VS_NEXT:

            *MAX_VS_NEXT* : float
                COMMENT:
                a two-element Numpy array containing the index of the element of
                `RESULT <LCAMechanism_OUTPUT.RESULT>` with the highest value (element 1) and the difference
                between that and the next highest one in `TRANSFER_RESULT` (element 2)
                COMMENT
                the difference between the two elements of the LCAMechanism's `value <Mechanism_Base.value>`
                with the highest values.

            .. _LCAMechanism_MAX_VS_AVG:

            *MAX_VS_AVG* : float
                COMMENT:
                a two-element Numpy array containing the index of the element of
                `RESULT` with the highest value (element 1) and the difference
                between that and the average of the value of all of `RESULT`'s
                other elements
                COMMENT
                the difference between the element of the LCAMechanism's `value <Mechanism_Base.value>`
                and the average of all of the other elements.

        """
        RESULT=RESULT
        MEAN=OUTPUT_MEAN
        MEDIAN=OUTPUT_MEDIAN
        STANDARD_DEVIATION=OUTPUT_STD_DEV
        VARIANCE=OUTPUT_VARIANCE
        ENERGY=ENERGY
        ENTROPY=ENTROPY
        MAX_VS_NEXT=MAX_VS_NEXT
        MAX_VS_AVG=MAX_VS_AVG
    # THIS WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
    # for item in [item[NAME] for item in DDM_standard_output_ports]:
    #     setattr(DDM_OUTPUT.__class__, item, item)


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class LCAMechanism(RecurrentTransferMechanism):
    """
    LCAMechanism(                    \
        default_variable=None,       \
        size=None,                   \
        function=Logistic,           \
        initial_value=None,          \
        leak=0.5,                    \
        competition=1.0,             \
        self_excitation=0.0,         \
        noise=0.0,                   \
        integrator_mode = True       \
        time_step_size = 0.1         \
        threshold = None             \
        threshold_criterion = VALUE  \
        clip=[float:min, float:max], \
        params=None,                 \
        name=None,                   \
        prefs=None)

    Subclass of `RecurrentTransferMechanism` that implements a Leaky Competitive Accumulator.

    The key distinguishing features of an LCAMechanism are:

    1. its `integrator_function <TransferMechanism.integrator_function>`,
       which implements the `LeakyCompetingIntegrator` (for which *rate* = *leak*)

    2. its `matrix <LCAMechanism.matrix>`, consisting of `self_excitation <LCAMechanism.self_excitation>` (diagonal)
       and `competition <LCAMechanism.competition>` (off diagonal) components.

    COMMENT:
        Description
        -----------
            LCAMechanism is a Subtype of the RecurrentTransferMechanism Subtype of the TransferMechanism
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
        `function <TransferMechanism.function>`, and the `primary OutputPort <OutputPort_Primary>`
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
        specifies the `rate <LeakyCompetingIntegrator.rate>` on the `LeakyCompetingIntegrator function
        <LeakyCompetingIntegrator>`, which scales the contribution of the `LeakyCompetingIntegrator's
        <LeakyCompetingIntegrator>` `previous_value <LeakyCompetingIntegrator.previous_value>` to the accumulation of
        the `LeakyCompetingIntegrator's value <LeakyCompetingIntegrator.value>` (:math:`x_{i}`) on each time step.
        See `LeakyCompetingIntegrator <LeakyCompetingIntegrator>` for more details on what the
        `LeakyCompetingIntegrator function  <LeakyCompetingIntegrator>` computes.

    competition : value : default 1.0
        specifies the magnitude of the off-diagonal terms in the LCAMechanism's recurrent projection, thereby scaling
        the contributions of the competing unit (all :math:`f(x)_{j}` where :math:`j \\neq i`) to the accumulation of
        the `LeakyCompetingIntegrator's value <LeakyCompetingIntegrator.value>` (:math:`x_{i}`) on each time step
        (see `LeakyCompetingIntegrator <LeakyCompetingIntegrator>` for more details on what the
        `LeakyCompetingIntegrator function  <LeakyCompetingIntegrator>` computes).

    self_excitation : value : default 0.0
        specifies the diagonal terms in the LCAMechanism's recurrent projection, thereby scaling the contributions of
        each unit's own recurrent value (:math:`f(x)_{i}`) to the accumulation of the `LeakyCompetingIntegrator's
        value <LeakyCompetingIntegrator.value>` (:math:`x_{i}`) on each time step. See `LeakyCompetingIntegrator
        <LeakyCompetingIntegrator>` for more details on what the `LeakyCompetingIntegrator function
        <LeakyCompetingIntegrator>` computes.

    threshold : float or None : default None
        specifes the value at which the Mechanism's `is_finished` attribute is set to True (see `threshold
        <LCAMechanism_Threshold>` for more details).

    threshold_criterion : VALUE, MAX_VS_NEXT, MAX_VS_AVG, str, or int : default VALUE
        specifies the criterion that is used to evaluate whether the threshold has been reached. If *MAX_VS_NEXT* or
        *MAX_VS_AVG* is specified, then the length of the LCAMCechanism's `value <Mechanism_Base.value>` must be at
        least 2 (see `threshold <LCAMechanism_Threshold>` for more details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <LCAMechanism Mechanism.name>`
        specifies the name of the LCAMechanism Mechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LCAMechanism Mechanism (see `prefs <LCAMechanism Mechanism.prefs>` for
        details).

    context : str : default ''componentType+INITIALIZNG''
           string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    matrix : 2d np.array
        the `matrix <MappingProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism, the
        `self_excitation <LCAMechanism.self_excitation>` attribute sets the values on the diagonal, and the
        `competition <LCAMechanism.competition>` attribute sets the magnitude of the negative off-diagonal values.

    leak : value : default 0.5
        sets the `rate <LeakyCompetingIntegrator.rate>` on the `LeakyCompetingIntegrator function
        <LeakyCompetingIntegrator>`, which scales the contribution of the `LeakyCompetingIntegrator's
        <LeakyCompetingIntegrator>` `previous_value <LeakyCompetingIntegrator.previous_value>` to the
        accumulation of the `LeakyCompetingIntegrator's value <LeakyCompetingIntegrator.value>`
        (:math:`x_{i}`) on each time step. See `LeakyCompetingIntegrator <LeakyCompetingIntegrator>`
        for more details on what the `LeakyCompetingIntegrator function  <LeakyCompetingIntegrator>` computes.

    competition : value : default 1.0
        sets the magnitude of the off-diagonal terms in the LCAMechanism's recurrent projection, thereby scaling the
        contributions of the competing unit (all :math:`f(x)_{j}` where :math:`j \\neq i`) to the accumulation of the
        `LeakyCompetingIntegrator's value <LeakyCompetingIntegrator.value>` (:math:`x_{i}`) on each time step. See
        `LeakyCompetingIntegrator <LeakyCompetingIntegrator>` for more details on what the `LeakyCompetingIntegrator
        function  <LeakyCompetingIntegrator>` computes.

    self_excitation : value : default 0.0
        sets the diagonal terms in the LCAMechanism's recurrent projection, thereby scaling the contributions of each
        unit's own recurrent value (:math:`f(x)_{i}`) to the accumulation of the `LeakyCompetingIntegrator's value
        <LeakyCompetingIntegrator.value>` (:math:`x_{i}`) on each time step. See `LeakyCompetingIntegrator
        <LeakyCompetingIntegrator>` for more details on what the `LeakyCompetingIntegrator function
        <LeakyCompetingIntegrator>` computes.

    name : str
        the name of the LCAMechanism Mechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LCAMechanism Mechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of LCAMechanism : LCAMechanism

    """
    componentType = LCA_MECHANISM

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        # RATE: None,
        BETA: None
    })

    class Parameters(RecurrentTransferMechanism.Parameters):
        """
            Attributes
            ----------

                competition
                    see `competition <LCAMechanism.competition>`

                    :default value: 1.0
                    :type: float

                function
                    see `function <LCAMechanism.function>`

                    :default value: `Logistic`
                    :type: `Function`

                initial_value
                    see `initial_value <LCAMechanism.initial_value>`

                    :default value: None
                    :type:

                integrator_mode
                    see `integrator_mode <LCAMechanism.integrator_mode>`

                    :default value: True
                    :type: bool

                leak
                    see `leak <LCAMechanism.leak>`

                    :default value: 0.5
                    :type: float

                self_excitation
                    see `self_excitation <LCAMechanism.self_excitation>`

                    :default value: 0.0
                    :type: float

                time_step_size
                    see `time_step_size <LCAMechanism.time_step_size>`

                    :default value: 0.1
                    :type: float
        """

        function = Parameter(Logistic, stateful=False, loggable=False)

        leak = Parameter(0.5, modulable=True)
        auto = Parameter(0.0, modulable=True, aliases='self_excitation')
        hetero = Parameter(-1.0, modulable=True)
        competition = Parameter(1.0, modulable=True)
        time_step_size = Parameter(0.1, modulable=True)

        initial_value = None
        integrator_mode = Parameter(True, setter=_integrator_mode_setter)

    standard_output_ports = RecurrentTransferMechanism.standard_output_ports.copy()
    standard_output_ports.extend([{NAME:MAX_VS_NEXT,
                                    FUNCTION:max_vs_next},
                                   {NAME:MAX_VS_AVG,
                                    FUNCTION:max_vs_avg}])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size:tc.optional(tc.any(int, list, np.array))=None,
                 input_ports:tc.optional(tc.any(list, dict))=None,
                 matrix=None,
                 function=Logistic,
                 initial_value=None,
                 leak=0.5,
                 competition=1.0,
                 hetero=None,
                 self_excitation=None,
                 noise=0.0,
                 integrator_mode=True,
                 time_step_size=0.1,
                 clip=None,
                 output_ports:tc.optional(tc.any(str, Iterable))=RESULT,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        """Instantiate LCAMechanism
        """

        # Default output_ports is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_ports is None or output_ports is RESULT:
            output_ports = [RESULT]

        if competition is not None and hetero is not None:
            if competition != -1.0 * hetero:
                raise LCAError(
                    'Both competition and hetero are specified. competition '
                    'must have the same magnitude but opposite sign of hetero. '
                    'Provided values: competition = {0} , hetero = {1}'.format(
                        competition,
                        hetero
                    )
                )
        elif competition is not None:
            hetero = -competition
        elif hetero is not None:
            competition = -hetero

        # MODIFIED 10/26/19 NEW: [JDC]
        # Implemented for backward compatibility or, if kept, ease of use
        termination_threshold, termination_measure, termination_comparison_op = self._parse_threshold_args(kwargs)
        # MODIFIED 10/26/19 END

        integrator_function = LeakyCompetingIntegrator

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(input_ports=input_ports,
                                                  leak=leak,
                                                  self_excitation=self_excitation,
                                                  hetero=hetero,
                                                  competition=competition,
                                                  integrator_mode=integrator_mode,
                                                  time_step_size=time_step_size,
                                                  output_ports=output_ports,
                                                  params=params)

        if not isinstance(self.standard_output_ports, StandardOutputPorts):
            self.standard_output_ports = StandardOutputPorts(self,
                                                               self.standard_output_ports,
                                                               indices=PRIMARY)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_ports=input_ports,
                         auto=self_excitation,
                         hetero=hetero,
                         function=function,
                         integrator_function=LeakyCompetingIntegrator,
                         initial_value=initial_value,
                         noise=noise,
                         clip=clip,
                         termination_threshold=termination_threshold,
                         termination_measure=termination_measure,
                         termination_comparison_op=termination_comparison_op,
                         output_ports=output_ports,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        # Do these here so that name of the object (assigned by super) can be used in the warning messages
        if matrix is not None:
            warnings.warn(f"The 'matrix' arg was specified for {self.name} but will not be used; "
                          f"the matrix for an {self.__class__.__name__} is specified using "
                          f"the 'self_excitation' and 'competition' args.")

        if competition < 0:
            warnings.warn(f"The 'competition' arg specified for {self.name} is a negative value ({competition}); "
                          f"note that this will result in a matrix that has positive off-diagonal elements "
                          f"since 'competition' is assumed to specify the magnitude of inhibition.")

    def _parse_threshold_args(self, kwargs):
        """Implements convenience arguments threshold and threshold_criterion

        These are translated into the appropriate specifications for the termination_threshold, termination_measure,
        and termination_comparison_op for TransferMechanism.

        Note:  specifying (threshold and termination_threshold) and/or (threshold and
        threshold_criterion and termination_measure) causes an error.
        """
        termination_threshold = kwargs.pop(TERMINATION_THRESHOLD, None)
        threshold = kwargs.pop('threshold', None)
        if threshold is not None:
            if termination_threshold is not None:
                raise LCAError(f"The {repr('threshold')} and {repr(TERMINATION_THRESHOLD)} "
                               f"args of {self.__name__} can't both be specified.")
            else:
                termination_threshold = threshold
        else:
            termination_threshold = termination_threshold or self.parameters.termination_threshold.default_value

        termination_measure = kwargs.pop(TERMINATION_MEASURE, None)
        termination_comparison_op = kwargs.pop(TERMINATION_COMPARISION_OP, None)
        threshold_criterion = kwargs.pop('threshold_criterion', None)
        if threshold_criterion is not None:
            if termination_measure is not None:
                raise LCAError(f"The {repr('threshold_criterion')} and {repr(TERMINATION_MEASURE)} "
                               f"args of {self.__name__} can't both be specified.")
            else:
                if threshold_criterion is VALUE:
                    termination_measure = lambda x: max(x)
                    termination_comparison_op = GREATER_THAN_OR_EQUAL
                elif threshold_criterion == MAX_VS_NEXT:
                    termination_measure = max_vs_next
                    termination_comparison_op = GREATER_THAN_OR_EQUAL
                elif threshold_criterion == MAX_VS_AVG:
                    termination_measure = max_vs_avg
                    termination_comparison_op = GREATER_THAN_OR_EQUAL
                else:
                    raise LCAError(f"Unrecognized value provided to 'threshold_criterion arg of {self.__name__}: "
                                   f"{threshold_criterion};  must be VALUE, MAX_VS_NEXT, or MAX_VS_AVG.")
        else:
            termination_measure = termination_measure or (lambda x: max(x))
            termination_comparison_op = termination_comparison_op or GREATER_THAN_OR_EQUAL

        return termination_threshold, termination_measure, termination_comparison_op

    def _get_integrated_function_input(self, function_variable, initial_value, noise, context):

        leak = self.get_current_mechanism_param("leak", context)
        time_step_size = self.get_current_mechanism_param("time_step_size", context)

        # if not self.integrator_function:
        if self.initialization_status == ContextFlags.INITIALIZING:
            self.integrator_function = LeakyCompetingIntegrator(
                function_variable,
                initializer=initial_value,
                noise=noise,
                time_step_size=time_step_size,
                rate=leak,
                owner=self)

        current_input = self.integrator_function._execute(
            function_variable,
            context=context,
            # Should we handle runtime params?
            runtime_params={
                INITIALIZER: initial_value,
                NOISE: noise,
                RATE: leak,
                TIME_STEP_SIZE: time_step_size
            },
        )

        return current_input
