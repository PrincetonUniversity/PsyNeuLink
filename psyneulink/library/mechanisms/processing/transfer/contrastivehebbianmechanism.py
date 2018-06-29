# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** ContrastiveHebbianMechanism *************************************************

"""
.. _ContrastiveHebbian_Overview:

Overview
--------

A ContrastiveHebbianMechanism is a subclass of `RecurrentTransferMechanism` that implements a single-layered recurrent
network and the Contrastive Hebbian learning rule.  See the following references for a description of the learning rule,
its relationship to the backpropagation learning rule, and its use in connectionist networks:

  `Movellan, J. R. (1991). Contrastive Hebbian learning in the continuous Hopfield model. In Connectionist Models
  (pp. 10-17) <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_

  `Xie, X., & Seung, H. S. (2003). Equivalence of backpropagation and contrastive Hebbian learning in a layered network.
  Neural computation, 15(2), 441-454 <https://www.mitpressjournals.org/doi/abs/10.1162/089976603762552988>`_

  `O'reilly, R. C. (2001). Generalization in interactive networks: The benefits of inhibitory competition and Hebbian
  learning. Neural computation, 13(6), 1199-1241 <https://www.mitpressjournals.org/doi/abs/10.1162/08997660152002834>`_

  `Verguts, T., & Notebaert, W. (2008). Hebbian learning of cognitive control: dealing with specific and nonspecific
  adaptation. Psychological review, 115(2), 518 <http://psycnet.apa.org/record/2008-04236-010>`_


.. _ContrastiveHebbian_Creation:

Creating a ContrastiveHebbianMechanism
-------------------------------------

A ContrastiveHebbianMechanism is created directly by calling its constructor.::

    import psyneulink as pnl
    my_linear_ContrastiveHebbian_mechanism = pnl.ContrastiveHebbianMechanism(function=pnl.Linear)
    my_logistic_ContrastiveHebbian_mechanism = pnl.ContrastiveHebbianMechanism(function=pnl.Logistic(gain=1.0,
                                                                                                    bias=-4.0))

The recurrent projection is automatically created using (1) the **matrix** argument or (2) the **auto** and **hetero**
arguments of the Mechanism's constructor, and is assigned to the mechanism's `recurrent_projection
<ContrastiveHebbianMechanism.recurrent_projection>` attribute.

If the **matrix** argument is used to create the recurrent projection, it must specify either a square matrix or an
`AutoAssociativeProjection` that uses one (the default is `HOLLOW_MATRIX`).::

    recurrent_mech_1 = pnl.ContrastiveHebbianMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=[[1.0, 2.0, 2.0],
                                                              [2.0, 1.0, 2.0],
                                                              [2.0, 2.0, 1.0]])

    recurrent_mech_2 = pnl.ContrastiveHebbianMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=pnl.AutoAssociativeProjection)

If the **auto** and **hetero** arguments are used to create the recurrent projection, they set the diagonal and
off-diagonal terms, respectively.::

    recurrent_mech_3 = pnl.ContrastiveHebbianMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      auto=1.0,
                                                      hetero=2.0)

.. note::

    In the examples above, recurrent_mech_1 and recurrent_mech_3 are identical.

In all other respects, a ContrastiveHebbianMechanism is specified in the same way as a standard `TransferMechanism`.

.. _ContrastiveHebbian_Learning:

Configuring Learning
~~~~~~~~~~~~~~~~~~~~

A ContrastiveHebbianMechanism can be configured for learning when it is created by assigning `True` to the
**enable_learning** argument of its constructor.  This creates an `AutoAssociativeLearningMechanism` that is used to
train its `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`, and assigns as its `function
<Function_Base.function>` the one  specified in the **learning_function** argument of the ContrastiveHebbianMechanism's
constructor.  By default, this is the `ContrastiveHebbian` Function;  however, it can be replaced by any other function
that is suitable for autoassociative learning;  that is, one that takes a list or 1d array of numeric values
(an "activity vector") and returns a 2d array or square matrix (the "weight change matrix") with the same dimensions
as the length of the activity vector. The AutoAssociativeLearningMechanism is assigned to the `learning_mechanism
<ContrastiveHebbianMechanism.learning_mechanism>` attribute and is used to modify the `matrix
<AutoAssociativeProjection.matrix>` parameter of its `recurrent_projection
<ContrastiveHebbianMechanism.recurrent_projection>` (also referenced by the ContrastiveHebbianMechanism's own `matrix
<ContrastiveHebbianMechanism.matrix>` parameter.

If a ContrastiveHebbianMechanism is created without configuring learning (i.e., **enable_learning** is assigned `False`
in its constructor -- the default value), then learning cannot be enabled for the Mechanism until it has been
configured for learning;  any attempt to do so will issue a warning and then be ignored.  Learning can be configured
once the Mechanism has been created by calling its `configure_learning <ContrastiveHebbianMechanism.configure_learning>`
method, which also enables learning.

.. _ContrastiveHebbian_Structure:

Structure
---------

The distinguishing feature of a ContrastiveHebbianMechanism is a self-projecting `AutoAssociativeProjection` -- that
is, one that projects from the Mechanism's `primary OutputState <OutputState_Primary>` back to its `primary
InputState <InputState_Primary>`.  This can be parameterized using its `matrix <ContrastiveHebbianMechanism.matrix>`,
`auto <ContrastiveHebbianMechanism.auto>`, and `hetero <ContrastiveHebbianMechanism.hetero>` attributes, and is
stored in its `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>` attribute.

A ContrastiveHebbianMechanism also has two additional `OutputStates <OutputState>:  an *ENERGY* OutputState and, if its
`function <ContrastiveHebbianMechanism.function>` is bounded between 0 and 1 (e.g., a `Logistic` function), an *ENTROPY*
OutputState.  Each of these report the respective values of the vector in it its *RESULTS* (`primary
<OutputState_Primary>`) OutputState.

Finally, if it has been `specified for learning <ContrastiveHebbian_Learning>`, the ContrastiveHebbianMechanism is
associated with an `AutoAssociativeLearningMechanism` that is used to train its `AutoAssociativeProjection`.
The `learning_enabled <ContrastiveHebbianMechanism.learning_enabled>` attribute indicates whether learning
is enabled or disabled for the Mechanism.  If learning was not configured when the Mechanism was created, then it cannot
be enabled until the Mechanism is `configured for learning <ContrastiveHebbian_Learning>`.

In all other respects the Mechanism is identical to a standard  `TransferMechanism`.

.. _ContrastiveHebbian_Execution:

Execution
---------

COMMENT:
  NOTE THAT IT IS ALWAYS RUN IN INTEGRATOR_MODE = TRUE
COMMENT

When a ContrastiveHebbianMechanism executes, its variable, as is the case with all mechanisms, is determined by the
projections the mechanism receives. This means that a ContrastiveHebbianMechanism's variable is determined in part by the
value of its own `primary OutputState <OutputState_Primary>` on the previous execution, and the `matrix` of the
recurrent projection.

COMMENT:
Previous version of sentence above: "When a ContrastiveHebbianMechanism executes, it includes in its input the value of
its `primary OutputState <OutputState_Primary>` from its last execution."
8/9/17 CW: Changed the sentence above. Rationale: If we're referring to the fact that the recurrent projection
takes the previous output before adding it to the next input, we should specifically mention the matrix transformation
that occurs along the way.

12/1/17 KAM: Changed the above to describe the ContrastiveHebbianMechanism's variable on this execution in terms of
projections received, which happens to include a recurrent projection from its own primary output state on the previous
execution
COMMENT

Like a `TransferMechanism`, the function used to update each element can be assigned using its `function
<ContrastiveHebbianMechanism.function>` parameter. It then transforms its input
(including from the recurrent projection) using the specified function and parameters (see `Transfer_Execution`),
and returns the results in its OutputStates.

If it has been `configured for learning <ContrastiveHebbian_Learning>`
and is executed as part of a `System`, then its associated `LearningMechanism` is executed during the `learning phase
<System_Learning>` of the `System's execution <System_Execution>`.

.. _ContrastiveHebbian_Class_Reference:

Class Reference
---------------

"""

import numbers
from collections import Iterable

import numpy as np
import typecheck as tc
from enum import IntEnum

from psyneulink.components.functions.function import Linear, is_function_type, ContrastiveHebbian
from psyneulink.components.mechanisms.adaptive.learning.learningmechanism import \
    ERROR_SIGNAL, LEARNING_SIGNAL, LearningMechanism
from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.globals.keywords import \
    ENERGY, ENTROPY, HOLLOW_MATRIX, MATRIX, MEAN, MEDIAN, NAME, CONTRASTIVE_HEBBIAN_MECHANISM, \
    RESULT, STANDARD_DEVIATION, VARIABLE, VARIANCE
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.library.mechanisms.adaptive.learning.autoassociativelearningmechanism import \
    AutoAssociativeLearningMechanism
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

__all__ = [
    'DECAY', 'CONTRASTIVE_HEBBIAN_OUTPUT', 'RecurrentTransferError', 'ContrastiveHebbianMechanism',
    'PLUS_PHASE_ACTIVITY', 'MINUS_PHASE_ACTIVITY'
]


PLUS_PHASE_ACTIVITY = 'plus_phase_activity_output'
MINUS_PHASE_ACTIVITY = 'minus_phase_activity_output'


class LearningPhase(IntEnum):
    MINUS = 1
    PLUS  = 0


# Used to index items of InputState.variable corresponding to recurrent and external inputs
INTERNAL = 0
EXTERNAL = 1


class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

DECAY = 'decay'

# This is a convenience class that provides list of standard_output_state names in IDE
class CONTRASTIVE_HEBBIAN_OUTPUT():

    """
        .. _ContrastiveHebbianMechanism_Standard_OutputStates:

        `Standard OutputStates <OutputState_Standard>` for
        `ContrastiveHebbianMechanism`

        .. TRANSFER_RESULT:

        *RESULT* : 1d np.array
            the result of the `function <ContrastiveHebbianMechanism.function>`
            of the Mechanism

        .. TRANSFER_MEAN:

        *MEAN* : float
            the mean of the result

        *VARIANCE* : float
            the variance of the result

        .. ENERGY:

        *ENERGY* : float
            the energy of the result, which is calculated using the `Stability` Function with the ``ENERGY`` metric

        .. ENTROPY:

        *ENTROPY* : float
            The entropy of the result, which is calculated using the `Stability` Function with the ENTROPY metric
            (Note: this is only present if the Mechanism's `function` is bounded
            between 0 and 1 (e.g. the `Logistic` Function)).

        .. PLUS_PHASE_ACTIVITY:

        *PLUS_PHASE_ACTIVITY* : 1d np.array
            The vector of activity at the end of the plus phase of a training trial.

        .. MINUS_PHASE_ACTIVITY:

        *MINUS_PHASE_ACTIVITY* : 1d np.array
            The vector of activity at the end of the minus phase of a training trial.
        """
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE
    ENERGY=ENERGY
    ENTROPY=ENTROPY
    PLUS_PHASE_ACTIVITY=PLUS_PHASE_ACTIVITY
    MINUS_PHASE_ACTIVITY=MINUS_PHASE_ACTIVITY


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class ContrastiveHebbianMechanism(RecurrentTransferMechanism):
    """
    ContrastiveHebbianMechanism(          \
    default_variable=None,                \
    size=None,                            \
    function=Linear,                      \
    matrix=HOLLOW_MATRIX,                 \
    auto=None,                            \
    hetero=None,                          \
    initial_value=None,                   \
    noise=0.0,                            \
    smoothing_factor=0.5,                 \
    clip=[float:min, float:max],          \
    learning_rate=None,                   \
    learning_function=ContrastiveHebbian, \
    params=None,                          \
    name=None,                            \
    prefs=None)

    Subclass of `TransferMechanism` that implements a single-layer auto-recurrent network.

    COMMENT:
        Description
        -----------
            ContrastiveHebbianMechanism is a Subtype of the TransferMechanism Subtype of the ProcessingMechanisms Type
            of the Mechanism Category of the Component class.
            It implements a TransferMechanism with a recurrent projection (default matrix: HOLLOW_MATRIX).
            In all other respects, it is identical to a TransferMechanism.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <ContrastiveHebbianMechanism.variable>` for
        `function <ContrastiveHebbianMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = ContrastiveHebbianMechanism(size = [3, 2])
            T2 = ContrastiveHebbian(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <ContrastiveHebbian_Structure>`,
        or an AutoAssociativeProjection to use.

        - If **auto** and **matrix** are both specified, the diagonal terms are determined by auto and the off-diagonal
          terms are determined by matrix.

        - If **hetero** and **matrix** are both specified, the diagonal terms are determined by matrix and the
          off-diagonal terms are determined by hetero.

        - If **auto**, **hetero**, and **matrix** are all specified, matrix is ignored in favor of auto and hetero.

    auto : number, 1D array, or None : default None
        specifies matrix as a diagonal matrix with diagonal entries equal to **auto**, if **auto** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**.

        In the following examples, assume that the default variable of the mechanism is length 4:

        - setting **auto** to 1 and **hetero** to -1 sets matrix to have a diagonal of
          1 and all non-diagonal entries -1:

            .. math::

                \\begin{bmatrix}
                    1 & -1 & -1 & -1 \\\\
                    -1 & 1 & -1 & -1 \\\\
                    -1 & -1 & 1 & -1 \\\\
                    -1 & -1 & -1 & 1 \\\\
                \\end{bmatrix}

        - setting **auto** to [1, 1, 2, 2] and **hetero** to -1 sets matrix to:

            .. math::

                \\begin{bmatrix}
                    1 & -1 & -1 & -1 \\\\
                    -1 & 1 & -1 & -1 \\\\
                    -1 & -1 & 2 & -1 \\\\
                    -1 & -1 & -1 & 2 \\\\
                \\end{bmatrix}

        - setting **auto** to [1, 1, 2, 2] and **hetero** to  [[3, 3, 3, 3], [3, 3, 3, 3], [4, 4, 4, 4], [4, 4, 4, 4]]
          sets matrix to:

            .. math::

                \\begin{bmatrix}
                    1 & 3 & 3 & 3 \\\\
                    3 & 1 & 3 & 3 \\\\
                    4 & 4 & 2 & 4 \\\\
                    4 & 4 & 4 & 2 \\\\
                \\end{bmatrix}

        See **matrix** for details on how **auto** and **hetero** may overwrite matrix.

        Can be modified by control.

    hetero : number, 2D array, or None : default None
        specifies matrix as a hollow matrix with all non-diagonal entries equal to **hetero**, if **hetero** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**.

        When diagonal entries of **hetero** are specified with non-zero values, these entries are set to zero before
        hetero is used to produce a matrix.

        See **hetero** (above) for details on how various **auto** and **hetero** specifications are summed to produce a
        matrix.

        See **matrix** (above) for details on how **auto** and **hetero** may overwrite matrix.

        Can be modified by control.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input.
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a value added to the result of the `function <ContrastiveHebbianMechanism.function>`. See `noise
        <ContrastiveHebbianMechanism.noise>` for more details.

    smoothing_factor : float : default 0.5
        the smoothing factor for exponential time averaging of input::

         result = (smoothing_factor * variable) +
         (1-smoothing_factor * input to mechanism's function on the previous time step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>` the item in
        index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to the
        value of `clip <ContrastiveHebbianMechanism.clip>` that it exceeds.


    enable_learning : boolean : default False
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <ContrastiveHebbianMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <ContrastiveHebbianMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <ContrastiveHebbianMechanism.learning_rate>` for details).

    learning_function : function : default ContrastiveHebbian
        specifies the function for the LearningMechanism if `learning has been specified
        <ContrastiveHebbian_Learning>` for the ContrastiveHebbianMechanism.  It can be any function so long as it
        takes a list or 1d array of numeric values as its `variable <Function_Base.variable>` and returns a sqaure
        matrix of numeric values with the same dimensions as the length of the input.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <ContrastiveHebbianMechanism.name>`
        specifies the name of the ContrastiveHebbianMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ContrastiveHebbianMechanism; see `prefs <ContrastiveHebbianMechanism.prefs>`
        for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <ContrastiveHebbianMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        back to its `primary inputState <Mechanism_InputStates>`.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input (only relevant if `smoothing_factor
        <ContrastiveHebbianMechanism.smoothing_factor>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    integrator_function:
        The `integrator_function <ContrastiveHebbianMechanism.integrator_function>` used by the Mechanism when it
        executes, which is an `AdaptiveIntegrator <AdaptiveIntegrator>`. Keep in mind that the `smoothing_factor
        <ContrastiveHebbianMechanism.smoothing_factor>` parameter of the `ContrastiveHebbianMechanism` corresponds to
        the `rate <ContrastiveHebbianMechanismIntegrator.rate>` of the `ContrastiveHebbianMechanismIntegrator`.

    COMMENT:
    ALWAYS TRUE;  MOVE THIS TO MODULE DOCSTRING
    integrator_mode:

        the variable of the mechanism is first passed into the following equation:

        .. math::
            value = previous\\_value(1-smoothing\\_factor) + variable \\cdot smoothing\\_factor + noise

        The result of the integrator function above is then passed into the `mechanism's function
        <ContrastiveHebbianMechanism.function>`. Note that on the first execution, *initial_value* sets previous_value.
    COMMENT

    noise : float or function : default 0.0
        value passed to the `integrator_function <ContrastiveHebbianMechanism.integrator_function>` that is added to
        the current input.

        If noise is a list or array, it must be the same length as `variable
        <ContrastiveHebbianMechanism.default_variable>`.

        If noise is specified as a single float or function, while `variable <ContrastiveHebbianMechanism.variable>`
        is a list or array, noise will be applied to each variable element. In the case of a noise function, this means
        that the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value
            from its distribution on each execution. If noise is specified as a float or as a function with a fixed
            output, then the noise will simply be an offset that remains the same across all executions.

    smoothing_factor : float : default 0.5
        the smoothing factor for exponential time averaging of input when::

          result = (smoothing_factor * current input) + (1-smoothing_factor * result on previous time_step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>`

        the item in index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the
        maximum allowable value; any element of the result that exceeds the specified minimum or maximum value is set
        to the value of `clip <ContrastiveHebbianMechanism.clip>` that it exceeds.

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    learning_enabled : bool : default False
        indicates whether learning has been enabled for the ContrastiveHebbianMechanism.  It is set to `True` if
        `learning is specified <ContrastiveHebbian_Learning>` at the time of construction (i.e., if the
        **enable_learning** argument of the Mechanism's constructor is assigned `True`, or when it is configured for
        learning using the `configure_learning <ContrastiveHebbianMechanism.configure_learning>` method.  Once learning
        has been configured, `learning_enabled <RecurrentMechahinsm.learning_enabled>` can be toggled at any time to
        enable or disable learning; however, if the Mechanism has not been configured for learning, an attempt to
        set `learning_enabled <RecurrentMechahinsm.learning_enabled>` to `True` elicits a warning and is then
        ignored.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        specifies the learning rate used by the `learning_function <ContrastiveHebbianMechanism.learning_function>`
        of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignement).

    learning_function : function : default ContrastiveHebbian
        the function used by the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>` if `learning is specified
        <ContrastiveHebbian_Learning>`.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <ContrastiveHebbian_Learning>`, and used to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.

    value : 2d np.array [array(float64)]
        result of executing `function <ContrastiveHebbianMechanism.function>`; same value as first item of
        `output_values <ContrastiveHebbianMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    output_states : Dict[str: OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:

        * `TRANSFER_RESULT`, the `value <OutputState.value>` of which is the **result** of `function
        <ContrastiveHebbianMechanism.function>`;
        * `TRANSFER_MEAN`, the `value <OutputState.value>` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the `value <OutputState.value>` of which is the variance of the result;
        * `ENERGY`, the `value <OutputState.value>` of which is the energy of the result,
          calculated using the `Stability` Function with the ENERGY metric;
        * `ENTROPY`, the `value <OutputState.value>` of which is the entropy of the result,
          calculated using the `Stability` Function with the ENTROPY metric;
          note:  this is only present if the Mechanism's `function <Mechanism_Base.function>` is bounded between 0 and 1
          (e.g., the `Logistic` function).
        * 'PLUS_PHASE_ACTIVITY', the `value <OutputState.value>` of which is the activity at the end of the plus
          phase of training.
        * 'MINUS_PHASE_ACTIVITY', the `value <OutputState.value>` of which is the activity at the end of the minus
          phase of training.

    output_values : List[array(float64), float, float]
        a list with the following items:

        * **result** of the ``function`` calculation (value of TRANSFER_RESULT OutputState);
        * **mean** of the result (``value`` of TRANSFER_MEAN OutputState)
        * **variance** of the result (``value`` of TRANSFER_VARIANCE OutputState);
        * **energy** of the result (``value`` of ENERGY OutputState);
        * **entropy** of the result (if the ENTROPY OutputState is present).
        * **plus_phase_activity** at the end of a training trial.
        * **minus_phase_activity** at the end of a training trial.

    name : str
        the name of the ContrastiveHebbianMechanism; if it is not specified in the **name** argument of the constructor,
        a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ContrastiveHebbianMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of ContrastiveHebbianMechanism : ContrastiveHebbianMechanism

    """
    componentType = CONTRASTIVE_HEBBIAN_MECHANISM

    class ClassDefaults(RecurrentTransferMechanism.ClassDefaults):
        variable = np.array([[0]])

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:PLUS_PHASE_ACTIVITY,
                                    VARIABLE:PLUS_PHASE_ACTIVITY},
                                   {NAME:MINUS_PHASE_ACTIVITY,
                                    VARIABLE:MINUS_PHASE_ACTIVITY}
                                   ])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict)) = None,
                 function=Linear,
                 matrix=HOLLOW_MATRIX,
                 auto=None,
                 hetero=None,
                 initial_value=None,
                 noise=0.0,
                 smoothing_factor: is_numeric_or_none=0.5,
                 clip=None,
                 enable_learning:bool=False,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = ContrastiveHebbian,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULT,
                 convergence_criterion:float=0.01,
                 # additional_output_states:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None):

        """Instantiate ContrastiveHebbianMechanism"""

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        # output_states = [PLUS_PHASE_ACTIVITY, MINUS_PHASE_ACTIVITY]
        # if additional_output_states:
        #     if isinstance(additional_output_states, list):
        #         output_states += additional_output_states
        #     else:
        #         output_states.append(additional_output_states)


        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(convergence_criterion=convergence_criterion,
                                                  # output_states=output_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         matrix=matrix,
                         auto=auto,
                         hetero=hetero,
                         has_recurrent_input_state=True,
                         initial_value=initial_value,
                         noise=noise,
                         integrator_mode=True,
                         smoothing_factor=smoothing_factor,
                         clip=clip,
                         enable_learning=enable_learning,
                         learning_rate=learning_rate,
                         learning_function=learning_function,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs)

    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)
        self.plus_phase_activity = None
        self.minus_phase_activity = None
        self.learning_phase = None

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    def _instantiate_learning_mechanism(self,
                                        activity_vector:tc.any(list, np.array),
                                        learning_function:tc.any(is_function_type),
                                        learning_rate:tc.any(numbers.Number, list, np.ndarray, np.matrix),
                                        matrix,
                                        context=None):

        objective_mechanism = ComparatorMechanism(sample=self.output_states[MINUS_PHASE_ACTIVITY],
                                                  target=self.output_states[PLUS_PHASE_ACTIVITY])

        learning_mechanism = AutoAssociativeLearningMechanism(default_variable=[objective_mechanism.value],
                                                              function=learning_function,
                                                              learning_rate=learning_rate,
                                                              name="{} for {}".format(
                                                                      AutoAssociativeLearningMechanism.className,
                                                                      self.name))

        # JDC: I DON'T THINK THESE ARE NEEDED (I THINK THEY ARE HANDLED AUTOMATICALLY BY THE CONSTRUCTOR
        #      FOR THE ComparatorMechanism) BUT I PUT THEM HERE JUST IN CASE THEY ARE NEEDED.
        # # Instantiate Projections from Mechanism's PLUS and MINUS PHASE OUTPUTS to ObjectiveMechanism
        # from psyneulink.globals.keywords import SAMPLE, TARGET
        # MappingProjection(sender=self.output_states[MINUS_PHASE_ACTIVITY],
        #                   receiver=objective_mechanism.input_states[SAMPLE],
        #                   name="Sample Projections for {}".format(objective_mechanism.name))
        # MappingProjection(sender=self.output_states[PLUS_PHASE_ACTIVITY],
        #                   receiver=objective_mechanism.input_states[TARGET],
        #                   name="Target Projections for {}".format(objective_mechanism.name))

        # Instantiate Projection from ObjectiveMechanism to LearningMechanism
        MappingProjection(sender=objective_mechanism,
                          receiver=learning_mechanism.input_states[ERROR_SIGNAL],
                          name="Learning Signal Projection for {}".format(learning_mechanism.name))

        # Instantiate Projection from LearningMechanism to Mechanism's AutoAssociativeProjection
        LearningProjection(sender=learning_mechanism.output_states[LEARNING_SIGNAL],
                           receiver=matrix.parameter_states[MATRIX],
                           name="{} for {}".format(LearningProjection.className, self.recurrent_projection.name))

        return learning_mechanism

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context=context)
        self.attributes_dict.update({PLUS_PHASE_ACTIVITY:self.plus_phase_activity,
                             MINUS_PHASE_ACTIVITY:self.minus_phase_activity})

    def _execute(self,
                 variable=None,
                 function_variable=None,
                 runtime_params=None,
                 context=None):

        if self.context.initialization_status == ContextFlags.INITIALIZING:
            return(variable)

        internal_input =  self.input_state.variable[INTERNAL]
        if self.context.flags_string == ContextFlags.EXECUTING:
            external_input = self.input_state.variable[EXTERNAL]
        else:
            external_input = self.input_state.socket_template

        if self.learning_phase is None:
            self.learning_phase = LearningPhase.PLUS

        if self.learning_phase == LearningPhase.PLUS:
            self.finished = False
            current_activity = external_input + internal_input
        else:
            current_activity = internal_input

        value = super()._execute(variable=current_activity,
                                 runtime_params=runtime_params,
                                 context=context)

        # Check for convergence
        if (value - self.integrator_function.previous_value) < self.convergence_criterion:

            # Terminate if this is the end of the minus phase
            if self.learning_phase == LearningPhase.MINUS:
                self.is_finished = True
                # JDC: NOT SURE THIS IS THE CORRECT THING TO DO
                self.input_state.variable[INTERNAL] = self.output_states[PLUS_PHASE_ACTIVITY].value

            # JDC: NOT SURE THIS IS THE CORRECT THING TO DO;  MAYBE ONLY AT BEGINNING OF MINUS PHASE?
            # NOTE: "socket_template" is a convenience property = np.zeros(<InputState>.variable.shape[-1])
            # Initialize internal input to zero for next phase
            self.input_state.variable[INTERNAL] = self.input_state.socket_template

            # Switch learning phase
            self.learning_phase = ~self.learning_phase

        return value
