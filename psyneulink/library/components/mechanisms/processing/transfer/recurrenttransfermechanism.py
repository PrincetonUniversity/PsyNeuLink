# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ****************************************  RecurrentTransferMechanism *************************************************

"""
.. _Recurrent_Transfer_Overview:

Overview
--------

A RecurrentTransferMechanism is a subclass of `TransferMechanism` that implements a single-layered recurrent
network, in which each element is connected to every other element (instantiated in a recurrent
`AutoAssociativeProjection` referenced by the Mechanism's `matrix <RecurrentTransferMechanism.matrix>` parameter).
Like a TransferMechanism, it can integrate its input prior to executing its `function
<RecurrentTransferMechanism.function>`. It can also report the energy and, if appropriate, the entropy of its output,
and can be configured to implement autoassociative (e.g., Hebbian) learning.

.. _Recurrent_Transfer_Creation:

Creating a RecurrentTransferMechanism
-------------------------------------

A RecurrentTransferMechanism is created directly by calling its constructor, for example::

    import psyneulink as pnl
    my_linear_recurrent_transfer_mechanism = pnl.RecurrentTransferMechanism(function=pnl.Linear)
    my_logistic_recurrent_transfer_mechanism = pnl.RecurrentTransferMechanism(function=pnl.Logistic(gain=1.0,
                                                                                                    bias=-4.0))

The recurrent projection is automatically created using (1) the **matrix** argument or (2) the **auto** and **hetero**
arguments of the Mechanism's constructor, and is assigned to the mechanism's `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` attribute.

If the **matrix** argument is used to create the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>`, it must specify either a square matrix or an
`AutoAssociativeProjection` that uses one (the default is `HOLLOW_MATRIX`).::

    recurrent_mech_1 = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=[[1.0, 2.0, 2.0],
                                                              [2.0, 1.0, 2.0],
                                                              [2.0, 2.0, 1.0]])

    recurrent_mech_2 = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=pnl.AutoAssociativeProjection)

If the **auto** and **hetero** arguments are used to create the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>`, they set the diagonal and off-diagonal terms, respectively.::

    recurrent_mech_3 = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      auto=1.0,
                                                      hetero=2.0)

.. note::

    In the examples above, recurrent_mech_1 and recurrent_mech_3 are identical.

In all other respects, a RecurrentTransferMechanism is specified in the same way as a standard `TransferMechanism`.

.. _Recurrent_Transfer_Learning:

*Configuring Learning*
~~~~~~~~~~~~~~~~~~~~~~

A RecurrentTransferMechanism can be configured for learning when it is created by assigning `True` to the
**enable_learning** argument of its constructor.  This creates an `AutoAssociativeLearningMechanism` that is used to
train its `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`, and assigns as its `function
<Function_Base.function>` the one  specified in the **learning_function** argument of the RecurrentTransferMechanism's
constructor.  By default, this is the `Hebbian` Function;  however, it can be replaced by any other function that is
suitable for autoassociative learning;  that is, one that takes a list or 1d array of numeric values
(an "activity vector") and returns a 2d array or square matrix (the "weight change matrix") with the same dimensions
as the length of the activity vector. The AutoAssociativeLearningMechanism is assigned to the `learning_mechanism
<RecurrentTransferMechanism.learning_mechanism>` attribute and is used to modify the `matrix
<AutoAssociativeProjection.matrix>` parameter of its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` (also referenced by the RecurrentTransferMechanism's own `matrix
<RecurrentTransferMechanism.matrix>` parameter.

If a RecurrentTransferMechanism is created without configuring learning (i.e., **enable_learning** is assigned `False`
in its constructor -- the default value), then learning cannot be enabled for the Mechanism until it has been
configured for learning;  any attempt to do so will issue a warning and then be ignored.  Learning can be configured
once the Mechanism has been created by calling its `configure_learning <RecurrentTransferMechanism.configure_learning>`
method, which also enables learning.

COMMENT:
8/7/17 CW: In past versions, the first sentence of the paragraph above was: "A RecurrentTransferMechanism can be
created directly by calling its constructor, or using the `mechanism() <Mechanism.mechanism>` command and specifying
RECURRENT_TRANSFER_MECHANISM as its **mech_spec** argument".
However, the latter method is no longer correct: it instead creates a DDM: the problem is line 590 in Mechanism.py,
as MechanismRegistry is empty!
10/9/17 MANTEL: mechanism() factory method is removed
COMMENT

.. _Recurrent_Transfer_Structure:

Structure
---------

The distinguishing feature of a RecurrentTransferMechanism is its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` attribute: a self-projecting `AutoAssociativeProjection`.
By default, `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` projects from the Mechanism's
`primary OutputState <OutputState_Primary>` back to its `primary InputState <InputState_Primary>`.  This can be
parameterized using its `matrix <RecurrentTransferMechanism.matrix>`, `auto <RecurrentTransferMechanism.auto>`,
and `hetero <RecurrentTransferMechanism.hetero>` attributes, and is stored in its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` attribute.  Using the `has_recurrent_input_state
<RecurrentTransferMechanism.has_recurrent_input_state>` attribute, the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` can also be made to project to a separate *RECURRENT* InputState
rather, than the primary one (named *EXTERNAL*).  In this case, the InputStates' results will be combined using the
`combination_function <RecurrentTransferMechanism.combination_function>` *before* being passed to the
RecurrentTransferMechanism's `function <RecurrentTransferMechanism.function>`.

A RecurrentTransferMechanism also has two additional `OutputStates <OutputState>`:  an *ENERGY* OutputState and, if its
`function <RecurrentTransferMechanism.function>` is bounded between 0 and 1 (e.g., a `Logistic` function), an *ENTROPY*
OutputState.  Each of these report the respective values of the vector in it its *RESULTS* (`primary
<OutputState_Primary>`) OutputState.

Finally, if it has been `specified for learning <Recurrent_Transfer_Learning>`, the RecurrentTransferMechanism is
associated with an `AutoAssociativeLearningMechanism` that is used to train its `AutoAssociativeProjection`.
The `learning_enabled <RecurrentTransferMechanism.learning_enabled>` attribute indicates whether learning
is enabled or disabled for the Mechanism.  If learning was not configured when the Mechanism was created, then it cannot
be enabled until the Mechanism is `configured for learning <Recurrent_Transfer_Learning>`.

In all other respects the Mechanism is identical to a standard  `TransferMechanism`.

.. _Recurrent_Transfer_Execution:

Execution
---------

When a RecurrentTransferMechanism executes, its variable, as is the case with all mechanisms, is determined by the
projections the mechanism receives. This means that a RecurrentTransferMechanism's variable is determined in part by
the value of its own `primary OutputState <OutputState_Primary>` on the previous execution, and the `matrix` of the
`recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

Like a `TransferMechanism`, the function used to update each element can be specified in the **function** argument
of its constructor.  It then transforms its input (including from the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>`) using the specified function and parameters (see
`Transfer_Execution`), and returns the results in its OutputStates.

Also like a `TransferMechanism`, the function used to integrate its input before passing it to `function
RecurrentTransferMechanism.function` (when `integrator_mode <RecurrentTransferMechanism.integrator_mode>` is `True`)
can be specified in the **integrator_function** argument of its constructor.

If a `convergence_criterion <RecurrentTransferMechanism.convergence_criterion>` is specified, then on each execution
the `convergence_function <RecurrentTransferMechanism.convergence_function>` is evaluated, and execution in the current
`trial` continues until the result returned is less than or equal to the `convergence_criterion
<RecurrentTransferMechanism.convergence_criterion>` or the number of executions reaches `max_passes
<RecurrentTransferMechanism.max_passes>` (if it is specified).

If it has been `configured for learning <Recurrent_Transfer_Learning>` and is executed as part of a `System`,
then its `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is executed when the `learning_condition
<RecurrentTransferMechanism.learning_condition>` is satisfied,  during the `execution phase <System_Execution>` of
the System's execution.  Note that this is distinct from the behavior of supervised learning algorithms (such as
`Reinforcement` and `BackPropagation`), that are executed during the `learning phase <System_Execution>` of a
System's execution.  By default, the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` executes,
and updates the `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` immediately after the
RecurrentTransferMechanism executes.

.. _Recurrent_Transfer_Class_Reference:

Class Reference
---------------

"""

import itertools
import numbers
import numpy as np
import typecheck as tc
import warnings

from collections.abc import Iterable
from types import MethodType

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import function_type, method_type
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.function import Function, is_function_type
from psyneulink.core.components.functions.learningfunctions import Hebbian
from psyneulink.core.components.functions.objectivefunctions import Distance, Stability
from psyneulink.core.components.functions.transferfunctions import Linear, get_matrix
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.adaptive.learning.learningmechanism import ACTIVATION_INPUT, LEARNING_SIGNAL, LearningMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.state import _instantiate_state
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import AUTO, ENERGY, ENTROPY, HETERO, HOLLOW_MATRIX, INPUT_STATE, MATRIX, MAX_ABS_DIFF, NAME, OUTPUT_MEAN, OUTPUT_MEDIAN, OUTPUT_STD_DEV, OUTPUT_VARIANCE, PARAMS_CURRENT, RECURRENT_TRANSFER_MECHANISM, RESULT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.registry import register_instance, remove_instance_from_registry
from psyneulink.core.globals.socket import ConnectionInfo
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.core.scheduling.condition import Condition, TimeScale, WhenFinished
from psyneulink.library.components.mechanisms.adaptive.learning.autoassociativelearningmechanism import AutoAssociativeLearningMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection, get_auto_matrix, get_hetero_matrix

__all__ = [
    'CONVERGENCE', 'EXTERNAL', 'EXTERNAL_INDEX',
    'RECURRENT', 'RECURRENT_INDEX', 'RECURRENT_OUTPUT', 'RecurrentTransferError', 'RecurrentTransferMechanism',
    'UPDATE'
]

EXTERNAL = 'EXTERNAL'
RECURRENT = 'RECURRENT'
# Used to index items of InputState.variable corresponding to recurrent and external inputs
EXTERNAL_INDEX = 0
RECURRENT_INDEX = -1

COMBINATION_FUNCTION = 'combination_function'

# Used to specify learning_condition
UPDATE = 'UPDATE'
CONVERGENCE = 'CONVERGENCE'


class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

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

        *OUTPUT_MEAN* : float
            the mean of the result

        *OUTPUT_VARIANCE* : float
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
    MEAN=OUTPUT_MEAN
    MEDIAN=OUTPUT_MEDIAN
    STANDARD_DEVIATION=OUTPUT_STD_DEV
    VARIANCE=OUTPUT_VARIANCE
    ENERGY=ENERGY
    ENTROPY=ENTROPY
    # THIS WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
    # for item in [item[NAME] for item in DDM_standard_output_states]:
    #     setattr(DDM_OUTPUT.__class__, item, item)


def _recurrent_transfer_mechanism_matrix_getter(owning_component=None, context=None):
    from psyneulink.library.components.projections.pathway.autoassociativeprojection import get_auto_matrix, get_hetero_matrix

    try:
        a = get_auto_matrix(owning_component.parameters.auto._get(context), owning_component.recurrent_size)
        c = get_hetero_matrix(owning_component.parameters.hetero._get(context), owning_component.recurrent_size)
        return a + c
    except TypeError:
        return None


def _get_auto_hetero_from_matrix(matrix):
    matrix = matrix.copy()
    auto = np.diag(matrix).copy()

    np.fill_diagonal(matrix, 0)
    hetero = matrix

    return auto, hetero


def _recurrent_transfer_mechanism_matrix_setter(value, owning_component=None, context=None):
    # KDM 8/3/18: This was attributed to a hack in how auto/hetero were implemented, but this behavior matches
    # the existing behavior. Unsure if this is actually correct though
    # KDM 8/7/18: removing the below because it has bad side effects for _instantiate_from_context, and it's not clear
    # that it's the correct behavior. Similar reason for removing/not implementing auto/hetero setters
    # if hasattr(owning_component, "recurrent_projection"):
    #     owning_component.recurrent_projection.parameter_states["matrix"].function.parameters.previous_value._set(value, base_execution_id)

    try:
        value = get_matrix(value, owning_component.recurrent_size, owning_component.recurrent_size)
    except AttributeError:
        pass

    if value is not None:
        auto, hetero = _get_auto_hetero_from_matrix(value)
        owning_component.parameters.auto._set(auto, context)
        owning_component.parameters.hetero._set(hetero, context)

    return value


def _recurrent_transfer_mechanism_learning_rate_setter(value, owning_component=None, context=None):
    if hasattr(owning_component, "learning_mechanism") and owning_component.learning_mechanism:
        owning_component.learning_mechanism.parameters.learning_rate._set(value, context)
    return value


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class RecurrentTransferMechanism(TransferMechanism):
    """
    RecurrentTransferMechanism(                         \
    default_variable=None,                              \
    size=None,                                          \
    function=Linear,                                    \
    matrix=HOLLOW_MATRIX,                               \
    auto=None,                                          \
    hetero=None,                                        \
    has_recurrent_input_state=False                     \
    combination_function=LinearCombination,             \
    integrator_mode=False,                              \
    integrator_function=AdaptiveIntegrator,             \
    initial_value=None,                                 \
    integration_rate=0.5,                               \
    noise=0.0,                                          \
    clip=[float:min, float:max],                        \
    convergence_function=Distance(metric=MAX_ABS_DIFF), \
    convergence_criterion=None,                         \
    max_passes=None,                                    \
    enable_learning=False,                              \
    learning_rate=None,                                 \
    learning_function=Hebbian,                          \
    learning_condition=UPDATE,                          \
    params=None,                                        \
    name=None,                                          \
    prefs=None)

    Subclass of `TransferMechanism` that implements a single-layer auto-recurrent network.

    COMMENT:
        Description
        -----------
            RecurrentTransferMechanism is a Subtype of the TransferMechanism Subtype of the ProcessingMechanisms Type
            of the Mechanism Category of the Component class.
            It implements a TransferMechanism with a recurrent projection (default matrix: HOLLOW_MATRIX).
            In all other respects, it is identical to a TransferMechanism.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <RecurrentTransferMechanism.variable>` for
        `function <RecurrentTransferMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
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

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <Recurrent_Transfer_Structure>`,
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

    has_recurrent_input_state : boolean : default False
        specifies whether the mechanism's `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
        points to a separate input state. By default, if False, the recurrent_projection points to its `primary
        InputState <InputState_Primary>`. If True, the recurrent_projection points to a separate input state, and
        the values of all input states are combined using `LinearCombination <function.LinearCombination>` *before*
        being passed to the RecurrentTransferMechanism's `function <RecurrentTransferMechanism.function>`.

    combination_function : function : default LinearCombination
        specifies function used to combine the *RECURRENT* and *INTERNAL* `InputStates <Recurrent_Transfer_Structure>`;
        must accept a 2d array with one or two items of the same length, and generate a result that is the same size
        as each of these;  default simply adds the two items.

    integrator_mode : bool : False
        specifies whether or not the RecurrentTransferMechanism should be executed using its `integrator_function
        <RecurrentTransferMechanism>` to integrate its `variable <RecurrentTransferMechanism.variable>` (
        when set to `True`), or simply report the asymptotic value of the output of its `function
        <RecurrentTransferMechanism.function>` (when set to `False`).

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use in `integration_mode <RecurrentTransferMechanism.integration_mode>`.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input if `integrator_mode
        <RecurrentTransferMechanism.integrator_mode>` is `True`).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    integration_rate : float : default 0.5
        the rate used for integrating `variable <RecurrentTransferMechanism.variable>` when `integrator_mode
        <RecurrentTransferMechanism.integrator_mode>` is set to `True`::

             result = (integration_rate * variable) +
             (1-integration_rate * input to mechanism's function on the previous time step)

    noise : float or function : default 0.0
        a value added to the result of the `function <RecurrentTransferMechanism.function>` or to the result of
        `integrator_function <RecurrentTransferMechanism.integrator_function>`, depending on whether `integrator_mode
        <RecurrentTransferMechanism.integrator_mode>` is `True` or `False`. See
        `noise <RecurrentTransferMechanism.noise>` for additional details.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <RecurrentTransferMechanism.function>` the item in
        index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to the
        value of `clip <RecurrentTransferMechanism.clip>` that it exceeds.

    convergence_function : function : default Distance(metric=MAX_ABS_DIFF)
        specifies the function that calculates `delta <RecurrentTransferMechanism.delta>`, and determines when
        `is_converged <RecurrentTransferMechanism.is_converged>` is `True`.

    convergence_criterion : float : default 0.01
        specifies the value of `delta <RecurrentTransferMechanism.delta>` at which `is_converged
        <RecurrentTransferMechanism.is_converged>` is `True`.

    max_passes : int : default 1000
        specifies maximum number of executions (`passes <TimeScale.PASS>`) that can occur in a trial before reaching
        the `convergence_criterion <RecurrentTransferMechanism.convergence_criterion>`, after which an error occurs;
        if `None` is specified, execution may continue indefinitely or until an interpreter exception is generated.

    enable_learning : boolean : default False
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <RecurrentTransferMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <RecurrentTransferMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <RecurrentTransferMechanism.learning_rate>` for details).

    learning_function : function : default Hebbian
        specifies the function for the LearningMechanism if `learning has been specified
        <Recurrent_Transfer_Learning>` for the RecurrentTransferMechanism.  It can be any function so long as it
        takes a list or 1d array of numeric values as its `variable <Function_Base.variable>` and returns a sqaure
        matrix of numeric values with the same dimensions as the length of the input.

    learning_condition : Condition, UPDATE, CONVERGENCE : default UPDATE
       specifies the `Condition` assigned to `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>`;
       A `Condition` can be used, or one of the following two keywords:

       * *UPDATE:* `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is executed immediately after
         every execution of the RecurrentTransferMechanism;  this is equivalent to assigning no `Condition`
       ..
       * *CONVERGENCE:* `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is executed whenever the
         the `convergence_criterion` is satisfied;  this is equivalent to a WhenFinished(``rec_mech``) `Condition`
         in which ``rec_mech`` is the RecurrentTransferMechanism.

       See `learning_condition <RecurrentTransferMechanism.learning_condition>` for additional details.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <RecurrentTransferMechanism.name>`
        specifies the name of the RecurrentTransferMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the RecurrentTransferMechanism; see `prefs <RecurrentTransferMechanism.prefs>`
        for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : 2d np.array with one item in axis 0.
        the input to Mechanism's `function <RecurrentTransferMechanism.function>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
         to its `primary InputState <Mechanism_InputStates>`.

    has_recurrent_input_state : boolean
        specifies whether the mechanism's `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
        points to a separate input state. If False, the recurrent_projection points to its `primary
        InputState <InputState_Primary>`. If True, the recurrent_projection points to a separate input state, and
        the values of all input states are combined using `LinearCombination <function.LinearCombination>` *before*
        being passed to the RecurrentTransferMechanism's `function <RecurrentTransferMechanism.function>`.

    combination_function : function
        the Function used to combine the *RECURRENT* and *EXTERNAL* InputStates if `has_recurrent_input_state
        <RecurrentTransferMechanism.has_recurrent_input_state>` is `True`.  By default this is a `LinearCombination`
        Function that simply adds them.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input (only relevant if `integration_rate
        <RecurrentTransferMechanism.integration_rate>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT






    integrator_mode :
        **When integrator_mode is set to True:**

        the variable of the mechanism is first passed into the following equation:

        .. math::
            value = previous\\_value(1-integration\\_rate) + variable \\cdot integration\\_rate + noise

        The result of the integrator function above is then passed into the mechanism's `function
        <RecurrentTransferMechanism.function>`. Note that on the first execution, *initial_value* determines the
        `integrator_function's <RecurrentTransferMechanism.integrator_function>` `previous_value
        <AdaptiveIntegrator.previous_value>`.

        **When integrator_mode is set to False:**

        The variable of the mechanism is passed into the `function of the mechanism
        <RecurrentTransferMechanism.function>`. The Mechanism's `integrator_function
        <RecurrentTransferMechanism.integrator_function>` is skipped entirely, and all related arguments
        (*noise*, *leak*, *initial_value*, and *time_step_size*) are ignored.

    integrator_mode : bool
        determines whether the RecurrentTransferMechanism uses its `integrator_function
        <RecurrentTransferMechanism.integrator_function>` to integrate its `variable
        <RecurrentTransferMechanism.variable>` when it executes.

        **If integrator_mode is set to** `True`:

            the RecurrentTransferMechanism's `variable <RecurrentTransferMechanism.variable>` is first passed to its
            `integrator_function <RecurrentTransferMechanism.integrator_function>`, and then the result is passed to
            its `function <RecurrentTransferMechanism.function>` which computes the RecurrentTransferMechanism's `value
            <RecurrentTransferMechanism.value>`.

            .. note::
                The RecurrentTransferMechanism's `integration_rate <RecurrentTransferMechanism.integration_rate>`,
                `noise <RecurrentTransferMechanism.noise>`, and `initial_value
                <RecurrentTransferMechanism.initial_value>` parameters specify the respective parameters of its
                `integrator_function <RecurrentTransferMechanism.integrator_function>` (with `initial_value
                <RecurrentTransferMechanism.initial_value>` corresponding to `initializer
                <IntegratorFunction.initializer>` and `integration_rate <RecurrentTransferMechanism.integration_rate>`
                corresponding to `rate <IntegratorFunction.rate>` of `integrator_function
                <RecurrentTransferMechanism.integrator_function>`). However, if there are any disagreements between
                these (e.g., any of these parameters is specified in the constructor for an `IntegratorFunction`
                assigned as the **integration_function** arg of the RecurrentTransferMechanism), the values specified
                for the `integrator_function <RecurrentTransferMechanism.integrator_function>` take precedence,
                and their value(s) are assigned as those of the corresponding parameters on the
                RecurrentTransferMechanism.

        **If integrator_mode is set to** `False`:

            if `noise <RecurrentTransferMechanism.noise>` is non-zero, it is applied to the RecurrentTransferMechanism's
            `variable <RecurrentTransferMechanism>` which is htne passed directly to its `function
            <RecurrentTransferMechanism.function>`  -- that is, its `integrator_function
            <RecurrentTransferMechanism.integrator_function>` is bypassed, and its related attributes (`initial_value
            <RecurrentTransferMechanism.initial_value>` and `integration_rate
            <RecurrentTransferMechanism.integration_rate>`) are ignored.

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <TransferMechanism.integrator_mode>` for details).

    integration_rate : float : default 0.5
        the rate used for integrating of `variable <RecurrentTransferMechanism.variable>` when `integrator_mode
        <RecurrentTransferMechanism.integrator_mode>` is set to `True`::

          result = (integration_rate * current input) + (1-integration_rate * result on previous time_step)

    noise : float or function : default 0.0
        When `integrator_mode <RecurrentTransferMechanism.integrator_mode>` is set to `True`, noise is passed into the
        `integrator_function <RecurrentTransferMechanism.integrator_function>`. Otherwise, noise is added to the result
        of the `function <RecurrentTransferMechanism.function>`.

        If noise is a list or array, it must be the same length as `variable
        <RecurrentTransferMechanism.default_variable>`.

        If noise is specified as a single float or function, while `variable <RecurrentTransferMechanism.variable>` is
        a list or array, noise will be applied to each variable element. In the case of a noise function, this means
        that the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value
            from its distribution on each execution. If noise is specified as a float or as a function with a fixed
            output, then the noise will simply be an offset that remains the same across all executions.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <RecurrentTransferMechanism.function>`

        the item in index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the
        maximum allowable value; any element of the result that exceeds the specified minimum or maximum value is set
        to the value of `clip <RecurrentTransferMechanism.clip>` that it exceeds.

    previous_value : 2d np.array [array(float64)] : default None
        `value <RecurrentTransferMechanism.value>` after the previous execution of the Mechanism; it is assigned `None`
        until the 2nd execution, and when the Mechanism's `reinitialize <Mechanism.reinitialize>` method is called.

        .. note::
           The RecurrentTransferMechanism's `previous_value` attribute is distinct from the `previous_value
           <AdaptiveIntegrator.previous_value>` attribute of its `integrator_function
           <RecurrentTransferMechanism.integrator_function>`.

    delta : scalar
        value returned by `convergence_function <RecurrentTransferMechanism.convergence_function>`;  used to determined
        when `is_converged <RecurrentTransferMechanism.is_converged>` is `True`.

    is_converged : bool
        `True` if `delta <RecurrentTransferMechanism.delta>` is less than or equal to `convergence_criterion
        <RecurrentTransferMechanism.convergence_criterion>`.

    convergence_function : function
        compares `value <RecurrentTransferMechanism.value>` with `previous_value
        <RecurrentTransferMechanism.previous_value>`; result is used to determine when `is_converged
        <RecurrentTransferMechanism.is_converged>` is `True`.

    convergence_criterion : float
        determines the value of `delta <RecurrentTransferMechanism.delta>` at which `is_converged
        <RecurrentTransferMechanism.is_converged>` is `True`.

    max_passes : int or None
        determines maximum number of executions (`passes <TimeScale.PASS>`) that can occur in a trial before reaching
        the `convergence_criterion <RecurrentTransferMechanism.convergence_criterion>`, after which an error occurs;
        if `None` is specified, execution may continue indefinitely or until an interpreter exception is generated.

    learning_enabled : bool : default False
        indicates whether learning has been enabled for the RecurrentTransferMechanism.  It is set to `True` if
        `learning is specified <Recurrent_Transfer_Learning>` at the time of construction (i.e., if the
        **enable_learning** argument of the Mechanism's constructor is assigned `True`, or when it is configured for
        learning using the `configure_learning <RecurrentTransferMechanism.configure_learning>` method.  Once learning
        has been configured, `learning_enabled <RecurrentMechahinsm.learning_enabled>` can be toggled at any time to
        enable or disable learning; however, if the Mechanism has not been configured for learning, an attempt to
        set `learning_enabled <RecurrentMechahinsm.learning_enabled>` to `True` elicits a warning and is then
        ignored.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <Recurrent_Transfer_Learning>`, and used to train the
        `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the `learning_function <RecurrentTransferMechanism.learning_function>`
        of the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignment).

    learning_function : function : default Hebbian
        the function used by the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` to train the
        `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` if `learning is specified
        <Recurrent_Transfer_Learning>`.

    learning_condition : Condition : default None
        determines the condition under which the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>`
        is executed in the context of a `Composition`; it can be specified in the **learning_condition** argument of
        the Mechanism's constructor or of its `configure_learning <RecurrentTransferMechanism.configure_learning>`
        method. By default, it executes immediately after the RecurrentTransferMechanism executes.

        .. note::
            The `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is an
            `AutoAssociativeLearningMechanism`, which executes during the `execution phase <System_Execution>`
            of the System's execution.  Note that this is distinct from the behavior of supervised learning algorithms
            (such as `Reinforcement` and `BackPropagation`), that are executed during the
            `learning phase <System_Execution>` of a System's execution

    value : 2d np.array [array(float64)]
        result of executing `function <RecurrentTransferMechanism.function>`; same value as first item of
        `output_values <RecurrentTransferMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    output_states : Dict[str: OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:

        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <RecurrentTransferMechanism.function>`;
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
        the name of the RecurrentTransferMechanism; if it is not specified in the **name** argument of the constructor,
        a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the RecurrentTransferMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    Returns
    -------
    instance of RecurrentTransferMechanism : RecurrentTransferMechanism

    """
    componentType = RECURRENT_TRANSFER_MECHANISM

    class Parameters(TransferMechanism.Parameters):
        """
            Attributes
            ----------

                auto
                    see `auto <RecurrentTransferMechanism.auto>`

                    :default value: 1
                    :type: int

                combination_function
                    see `combination_function <RecurrentTransferMechanism.combination_function>`

                    :default value: `LinearCombination`
                    :type: `Function`

                convergence_function
                    see `convergence_function <RecurrentTransferMechanism.convergence_function>`

                    :default value: `Distance`(metric=max_abs_diff)
                    :type: `Function`

                enable_learning
                    see `enable_learning <RecurrentTransferMechanism.enable_learning>`

                    :default value: False
                    :type: bool

                hetero
                    see `hetero <RecurrentTransferMechanism.hetero>`

                    :default value: 0
                    :type: int

                integration_rate
                    see `integration_rate <RecurrentTransferMechanism.integration_rate>`

                    :default value: 0.5
                    :type: float

                learning_condition
                    see `learning_condition <RecurrentTransferMechanism.learning_condition>`

                    :default value: None
                    :type:

                learning_function
                    see `learning_function <RecurrentTransferMechanism.learning_function>`

                    :default value: `Hebbian`
                    :type: `Function`

                learning_rate
                    see `learning_rate <RecurrentTransferMechanism.learning_rate>`

                    :default value: None
                    :type:

                matrix
                    see `matrix <RecurrentTransferMechanism.matrix>`

                    :default value: `HOLLOW_MATRIX`
                    :type: str

                noise
                    see `noise <RecurrentTransferMechanism.noise>`

                    :default value: 0.0
                    :type: float

                smoothing_factor
                    see `smoothing_factor <RecurrentTransferMechanism.smoothing_factor>`

                    :default value: 0.5
                    :type: float

        """
        matrix = Parameter(HOLLOW_MATRIX, modulable=True, getter=_recurrent_transfer_mechanism_matrix_getter, setter=_recurrent_transfer_mechanism_matrix_setter)
        auto = Parameter(1, modulable=True)
        hetero = Parameter(0, modulable=True)
        combination_function = LinearCombination
        integration_rate = Parameter(0.5, modulable=True)
        convergence_function = Distance(metric=MAX_ABS_DIFF)
        noise = Parameter(0.0, modulable=True)
        smoothing_factor = Parameter(0.5, modulable=True)
        enable_learning = False
        learning_function = Parameter(Hebbian, stateful=False, loggable=False)
        learning_rate = Parameter(None, modulable=True, setter=_recurrent_transfer_mechanism_learning_rate_setter)
        learning_condition = Parameter(None, stateful=False, loggable=False)

    paramClassDefaults = TransferMechanism.paramClassDefaults.copy()

    standard_output_states = TransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:ENERGY}, {NAME:ENTROPY}])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict)) = None,
                 has_recurrent_input_state=False,
                 combination_function:is_function_type=LinearCombination,
                 function=Linear,
                 matrix=None,
                 auto=None,
                 hetero=None,
                 integrator_mode=False,
                 integrator_function=AdaptiveIntegrator,
                 initial_value=None,
                 integration_rate: is_numeric_or_none=0.5,
                 noise=0.0,
                 clip=None,
                 convergence_function:tc.any(is_function_type)=Distance(metric=MAX_ABS_DIFF),
                 convergence_criterion:float=0.01,
                 max_passes:tc.optional(int)=1000,
                 enable_learning:bool=False,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = Hebbian,
                 learning_condition:tc.optional(tc.any(Condition, TimeScale,
                                                       tc.enum(UPDATE, CONVERGENCE)))=None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULT,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None,
                 **kwargs):
        """Instantiate RecurrentTransferMechanism
        """

        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None or output_states is RESULT:
            output_states = [RESULT]

        if isinstance(hetero, (list, np.matrix)):
            hetero = np.array(hetero)

        if isinstance(auto, list):
            auto = np.array(auto)

        # since removing the default argument matrix=HOLLOW_MATRIX to detect a user setting,
        # some hidden steps in validate_params that set this case to HOLLOW_MATRIX did not
        # happen
        if matrix is AutoAssociativeProjection:
            matrix = HOLLOW_MATRIX

        self._learning_enabled = enable_learning

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                  integrator_mode=integrator_mode,
                                                  learning_rate=learning_rate,
                                                  learning_function=learning_function,
                                                  learning_condition=learning_condition,
                                                  auto=auto,
                                                  hetero=hetero,
                                                  has_recurrent_input_state=has_recurrent_input_state,
                                                  combination_function=combination_function,
                                                  output_states=output_states,
                                                  params=params,
                                                  )

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         integrator_function=integrator_function,
                         initial_value=initial_value,
                         noise=noise,
                         integrator_mode=integrator_mode,
                         integration_rate=integration_rate,
                         clip=clip,
                         convergence_function=convergence_function,
                         convergence_criterion=convergence_criterion,
                         max_passes=max_passes,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    # def _handle_default_variable(self, default_variable=None, size=None, input_states=None, params=None):
    #     """Set self.recurrent_size if it was not set by subclass;  assumes it is size of first item"""
    #     default_variable = super()._handle_default_variable(default_variable, size, input_states, params)
    #     self.recurrent_size = self.recurrent_size or len(default_variable[0])
    #     return default_variable

    def _instantiate_defaults(
            self,variable=None,request_set=None,assign_missing=True,target_set=None,default_set=None,context=None):
        """Set self.recurrent_size if it was not set by subclass;  assumes it is size of first item of variable"""
        try:
            self.recurrent_size
        except AttributeError:
            self.recurrent_size = len(variable[0])
        super()._instantiate_defaults(variable,request_set,assign_missing,target_set,default_set, context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of auto, hetero, matrix.
        """
        from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # KDM 10/24/18: rearranged matrix and auto/hetero validation to correspond with comment/code in __init__
        # on this same date.
        # Validate MATRIX
        if MATRIX in target_set:

            matrix_param = target_set[MATRIX]

            if isinstance(matrix_param, AutoAssociativeProjection):
                matrix = matrix_param.matrix

            elif isinstance(matrix_param, str):
                matrix = get_matrix(matrix_param, rows=self.recurrent_size, cols=self.recurrent_size)

            elif isinstance(matrix_param, (np.matrix, list)):
                matrix = np.array(matrix_param)

            else:
                matrix = matrix_param
            if matrix is None:
                rows = cols = self.recurrent_size # this is a hack just to skip the tests ahead:
                # if the matrix really is None, that is checked up ahead, in _instantiate_attributes_before_function()
            else:
                rows = np.array(matrix).shape[0]
                cols = np.array(matrix).shape[1]

            try:
                if 'U' in repr(matrix.dtype):
                    raise RecurrentTransferMechanism("{0} has non-numeric entries".format(matrix))
            except AttributeError:
                pass

            # Shape of matrix must be square
            if rows != cols:
                if isinstance(matrix_param, AutoAssociativeProjection):
                    # if __name__ == '__main__':
                    err_msg = ("{} param of {} must be square to be used as recurrent projection for {}".
                               format(MATRIX, matrix_param.name, self.name))
                else:
                    err_msg = "{0} param for {1} must be square; currently, the {0} param is: {2}".\
                        format(MATRIX, self.name, matrix)
                raise RecurrentTransferError(err_msg)

            # Size of matrix must equal length of variable:
            if rows != self.recurrent_size:
                if isinstance(matrix_param, AutoAssociativeProjection):
                    err_msg = ("Number of rows in {} param for {} ({}) must be same as the size of variable for "
                               "{} {} (whose size is {} and whose variable is {})".
                               format(MATRIX, self.name, rows, self.__class__.__name__, self.name, self.size,
                                      self.defaults.variable))
                else:
                    err_msg = ("Size of {} param for {} ({}) must be the same as its variable ({})".
                               format(MATRIX, self.name, rows, self.recurrent_size))
                raise RecurrentTransferError(err_msg)


        if AUTO in target_set:
            auto_param = target_set[AUTO]
            if (auto_param is not None) and not isinstance(auto_param, (np.ndarray, list, numbers.Number)):
                raise RecurrentTransferError("auto parameter ({}) of {} is of incompatible type: it should be a "
                                             "number, None, or a 1D numeric array".format(auto_param, self))
            if isinstance(auto_param, (np.ndarray, list)) and len(auto_param) != 1 and len(auto_param) != self.size[0]:
                raise RecurrentTransferError("auto parameter ({0}) for {1} is of incompatible length with the size "
                                             "({2}) of its owner, {1}.".format(auto_param, self, self.size[0]))

        if HETERO in target_set:
            hetero_param = target_set[HETERO]
            if hetero_param is not None and not isinstance(hetero_param, (np.matrix, np.ndarray, list, numbers.Number)):
                raise RecurrentTransferError("hetero parameter ({}) of {} is of incompatible type: it should be a "
                                             "number, None, or a 2D numeric matrix or array".format(hetero_param, self))
            hetero_shape = np.array(hetero_param).shape
            if hetero_shape != (1,) and hetero_shape != (1, 1):
                if isinstance(hetero_param, (np.ndarray, list, np.matrix)) and hetero_shape[0] != self.size[0]:
                    raise RecurrentTransferError("hetero parameter ({0}) for {1} is of incompatible size with the size "
                                                 "({2}) of its owner, {1}.".format(hetero_param, self, self.size[0]))
                if isinstance(hetero_param, (np.ndarray, list, np.matrix)) and hetero_shape[0] != hetero_shape[1]:
                    raise RecurrentTransferError("hetero parameter ({}) for {} must be square.".format(hetero_param, self))

        # Validate DECAY
        # if DECAY in target_set and target_set[DECAY] is not None:
        #
        #     decay = target_set[DECAY]
        #     if not (0.0 <= decay and decay <= 1.0):
        #         raise RecurrentTransferError("{} argument for {} ({}) must be from 0.0 to 1.0".
        #                                      format(DECAY, self.name, decay))

        # FIX: validate learning_function and learning_rate here (use Hebbian as template for learning_rate

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """using the `matrix` argument the user passed in (which is now stored in function_params), instantiate
        ParameterStates for auto and hetero if they haven't already been instantiated. This is useful if auto and
        hetero were None in the initialization call.
        :param function:
        """
        self.parameters.previous_value._set(None, context)

        super()._instantiate_attributes_before_function(function=function, context=context)

        param_keys = self._parameter_states.key_values

        matrix = get_matrix(self.defaults.matrix, rows=self.recurrent_size, cols=self.recurrent_size)

        # below implements the rules provided by KAM:
        # - If auto and hetero but not matrix are specified, the diagonal terms of the matrix are determined by auto and the off-diagonal terms are determined by hetero.
        # - If auto, hetero, and matrix are all specified, matrix is ignored in favor of auto and hetero.
        # - If auto and matrix are both specified, the diagonal terms are determined by auto and the off-diagonal terms are determined by matrix. ​
        # - If hetero and matrix are both specified, the diagonal terms are determined by matrix and the off-diagonal terms are determined by hetero.
        auto = get_auto_matrix(self.defaults.auto, self.recurrent_size)
        hetero = get_hetero_matrix(self.defaults.hetero, self.recurrent_size)
        auto_specified = self.parameters.auto._user_specified
        hetero_specified = self.parameters.hetero._user_specified

        if auto_specified and hetero_specified:
            matrix = auto + hetero
        elif auto_specified:
            np.fill_diagonal(matrix, 0)
            matrix = matrix + auto
        elif hetero_specified:
            diag = np.diag(matrix)
            matrix = hetero.copy()
            np.fill_diagonal(matrix, diag)

        self.parameters.matrix._set(matrix, context)

        # 9/23/17 JDC: DOESN'T matrix arg default to something?
        # If no matrix was specified, then both AUTO and HETERO must be specified
        if (
            matrix is None
            and (
                AUTO not in param_keys
                or HETERO not in param_keys
                or not auto_specified
                or not hetero_specified
            )
        ):
            raise RecurrentTransferError("Matrix parameter ({}) for {} failed to produce a suitable matrix: "
                                         "if the matrix parameter does not produce a suitable matrix, the "
                                         "'auto' and 'hetero' parameters must be specified; currently, either"
                                         "auto or hetero parameter is missing.".format(self.params[MATRIX], self))

        if AUTO not in param_keys and HETERO in param_keys:
            d = np.diagonal(matrix).copy()
            state = _instantiate_state(owner=self,
                                       state_type=ParameterState,
                                       name=AUTO,
                                       reference_value=d,
                                       reference_value_name=AUTO,
                                       params=None,
                                       context=context)
            self._auto = d
            if state is not None:
                self._parameter_states[AUTO] = state
                state.source = self
            else:
                raise RecurrentTransferError("Failed to create ParameterState for `auto` attribute for {} \"{}\"".
                                           format(self.__class__.__name__, self.name))
        if HETERO not in param_keys and AUTO in param_keys:

            m = matrix.copy()
            np.fill_diagonal(m, 0.0)
            self._hetero = m
            state = _instantiate_state(owner=self,
                                       state_type=ParameterState,
                                       name=HETERO,
                                       reference_value=m,
                                       reference_value_name=HETERO,
                                       params=None,
                                       context=context)
            if state is not None:
                self._parameter_states[HETERO] = state
                state.source = self
            else:
                raise RecurrentTransferError("Failed to create ParameterState for `hetero` attribute for {} \"{}\"".
                                           format(self.__class__.__name__, self.name))

        if self.has_recurrent_input_state:
            comb_fct = self.combination_function
            if (
                not (
                    isinstance(comb_fct, LinearCombination)
                    or (isinstance(comb_fct, type) and issubclass(comb_fct, LinearCombination))
                    or (isinstance(comb_fct, MethodType) and comb_fct.__self__ == self)
                )
            ):
                if isinstance(comb_fct, type):
                    comb_fct = comb_fct()
                elif isinstance(comb_fct, (function_type, method_type)):
                    comb_fct = UserDefinedFunction(comb_fct, self.defaults.variable)
                try:
                    cust_fct_result = comb_fct.execute(self.defaults.variable)
                except AssertionError:
                    raise RecurrentTransferError(
                        "Function specified for {} argument of {} ({}) does not take an array with two items ({})".format(
                            repr(COMBINATION_FUNCTION),
                            self.name,
                            comb_fct,
                            self.defaults.variable
                        )
                    )
                try:
                    assert len(cust_fct_result) == len(self.defaults.variable[0])
                except AssertionError:
                    raise RecurrentTransferError(
                        "Function specified for {} argument of {} ({}) did not return a result that is"
                        " the same shape as the input to {} ({})".format(
                            repr(COMBINATION_FUNCTION),
                            self.name,
                            comb_fct,
                            self.name,
                            self.defaults.variable[0]
                        )
                    )

            # If combination_function is a method of a subclass, let it pass
            if not isinstance(comb_fct, Function):
                if isinstance(comb_fct, type):
                    self.combination_function = comb_fct(default_variable=self.defaults.variable)
                elif isinstance(comb_fct, MethodType) and comb_fct.__self__ == self:
                    pass
                else:
                    self.combination_function = UserDefinedFunction(custom_function=comb_fct,
                                                                     default_variable=self.defaults.variable)
            else:
                self.combination_function = comb_fct

        else:
            self.combination_function = None

        if self.auto is None and self.hetero is None:
            self.matrix = matrix
            if self.matrix is None:
                raise RecurrentTransferError("PROGRAM ERROR: Failed to instantiate \'matrix\' param for {}".
                                             format(self.__class__.__name__))

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate recurrent_projection, matrix, and the functions for the ENERGY and ENTROPY OutputStates
        """
        from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

        super()._instantiate_attributes_after_function(context=context)

        # (7/19/17 CW) this line of code is now questionable, given the changes to matrix and the recurrent projection
        if isinstance(self.matrix, AutoAssociativeProjection):
            self.recurrent_projection = self.matrix

        # IMPLEMENTATION NOTE:  THESE SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
        else:
            self.recurrent_projection = self._instantiate_recurrent_projection(self,
                                                                               matrix=self.matrix,
                                                                               context=context)
        self.aux_components.append(self.recurrent_projection)

        if self.learning_enabled:
            self.configure_learning(context=context)

        if ENERGY in self.output_states.names:
            energy = Stability(self.defaults.variable[0],
                               metric=ENERGY,
                               transfer_fct=self.function,
                               matrix=self.recurrent_projection._parameter_states[MATRIX])
            self.output_states[ENERGY]._calculate = energy.function

        if ENTROPY in self.output_states.names:
            if self.function.bounds == (0,1) or self.clip == (0,1):
                entropy = Stability(self.defaults.variable[0],
                                    metric=ENTROPY,
                                    transfer_fct=self.function,
                                    matrix=self.recurrent_projection._parameter_states[MATRIX])
                self.output_states[ENTROPY]._calculate = entropy.function
            else:
                del self.output_states[ENTROPY]

    def _update_parameter_states(self, context=None, runtime_params=None):
        for state in self._parameter_states:
            # (8/2/17 CW) because the auto and hetero params are solely used by the AutoAssociativeProjection
            # (the RecurrentTransferMechanism doesn't use them), the auto and hetero param states are updated in the
            # projection's _update_parameter_states, and accordingly are not updated here
            if state.name != AUTO and state.name != HETERO:
                state._update(context=context, params=runtime_params)

    def _update_previous_value(self, context=None):
        value = self.parameters.value._get(context)
        if value is None:
            value = self.defaults.value
        self.parameters.previous_value._set(value, context)

    # 8/2/17 CW: this property is not optimal for performance: if we want to optimize performance we should create a
    # single flag to check whether to get matrix from auto and hetero?
    @property
    def matrix(self):
        return self.parameters.matrix._get(self.most_recent_context)

    @matrix.setter
    def matrix(self, val): # simplified version of standard setter (in Component.py)
        # KDM 10/12/18: removing below because it doesn't seem to be correct, and also causes
        # unexpected values to be set to previous_value
        # KDM 7/1/19: reinstating below
        if hasattr(self, "recurrent_projection"):
            self.recurrent_projection.parameter_states["matrix"].function.previous_value = val

        self.parameters.matrix._set(val, self.most_recent_context)

        if hasattr(self, '_parameter_states') and 'matrix' in self._parameter_states:
            param_state = self._parameter_states['matrix']

            if hasattr(param_state.function, 'initializer'):
                param_state.function.reinitialize = val

    @property
    def auto(self):
        return self.parameters.auto._get(self.most_recent_context)

    @auto.setter
    def auto(self, val):
        self.parameters.auto._set(val, self.most_recent_context)

        if hasattr(self, "recurrent_projection") and 'hetero' in self._parameter_states:
            self.recurrent_projection.parameter_states["matrix"].function.previous_value = self.matrix


    @property
    def hetero(self):
        return self.parameters.hetero._get(self.most_recent_context)

    @hetero.setter
    def hetero(self, val):
        self.parameters.hetero._set(val, self.most_recent_context)

        if hasattr(self, "recurrent_projection") and 'auto' in self._parameter_states:
            self.recurrent_projection.parameter_states["matrix"].function.previous_value = self.matrix_param

    @property
    def learning_enabled(self):
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value:bool):

        self._learning_enabled = value
        # Enable learning for RecurrentTransferMechanism's learning_mechanism
        if hasattr(self, 'learning_mechanism'):
            self.learning_mechanism.learning_enabled = value
        # If RecurrentTransferMechanism has no LearningMechanism, warn and then ignore attempt to set learning_enabled
        elif value is True:
            warnings.warn("Learning cannot be enabled for {} because it has no {}".
                  format(self.name, LearningMechanism.__name__))
            return

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    @tc.typecheck
    def _instantiate_recurrent_projection(self,
                                          mech: Mechanism_Base,
                                          # this typecheck was failing, I didn't want to fix (7/19/17 CW)
                                          # matrix:is_matrix=HOLLOW_MATRIX,
                                          matrix=HOLLOW_MATRIX,
                                          context=None):
        """Instantiate an AutoAssociativeProjection from Mechanism to itself

        """

        from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection
        if isinstance(matrix, str):
            size = len(mech.defaults.variable[0])
            matrix = get_matrix(matrix, size, size)

        # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
        if self.has_recurrent_input_state:
            # # FIX: 7/12/18 MAKE THIS A METHOD THAT CAN BE OVERRIDDEN BY CONTRASTIVEHEBBIAN
            new_input_state = InputState(owner=self, name=RECURRENT, variable=self.defaults.variable[0],
                                         internal_only=True)
            assert (len(new_input_state.all_afferents) == 0)  # just a sanity check
            assert(self.input_state.name != "Recurrent Input State")
            # Rename existing InputState as EXTERNAL
            remove_instance_from_registry(registry=self._stateRegistry,
                                          category=INPUT_STATE,
                                          component=self.input_state)
            register_instance(self.input_state, EXTERNAL, InputState, self._stateRegistry, INPUT_STATE)
            proj = AutoAssociativeProjection(owner=mech,
                                             receiver=new_input_state,
                                             matrix=matrix,
                                             name=mech.name + ' recurrent projection')
            receiver = new_input_state
        else:
            proj = AutoAssociativeProjection(owner=mech,
                                             matrix=matrix,
                                             name=mech.name + ' recurrent projection')
            receiver = self.input_state

        proj._activate_for_compositions(ConnectionInfo.ALL)
        return proj

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    def _instantiate_learning_mechanism(self,
                                        activity_vector:tc.any(list, np.array),
                                        learning_function,
                                        learning_rate,
                                        learning_condition,
                                        matrix,
                                        context=None):

        learning_mechanism = AutoAssociativeLearningMechanism(default_variable=[activity_vector.value],
                                                              # learning_signals=[self.recurrent_projection],
                                                              function=learning_function,
                                                              learning_rate=learning_rate,
                                                              name="{} for {}".format(
                                                                      AutoAssociativeLearningMechanism.className,
                                                                      self.name))
        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        # (via aux_components attr)

        learning_mechanism.condition = learning_condition
        self.aux_components.append(learning_mechanism)
        # Instantiate Projection from Mechanism's output to LearningMechanism
        mproj = MappingProjection(sender=activity_vector,
                          receiver=learning_mechanism.input_states[ACTIVATION_INPUT],
                          name="Error Projection for {}".format(learning_mechanism.name))
        mproj._activate_for_all_compositions()
        self.aux_components.append(mproj)
        # Instantiate Projection from LearningMechanism to Mechanism's AutoAssociativeProjection
        lproj = LearningProjection(sender=learning_mechanism.output_states[LEARNING_SIGNAL],
                           receiver=matrix.parameter_states[MATRIX],
                           name="{} for {}".format(LearningProjection.className, self.recurrent_projection.name))
        lproj._activate_for_all_compositions()
        self.aux_components.append((lproj, True))
        return learning_mechanism

    @handle_external_context()
    def configure_learning(self,
                           learning_function:tc.optional(tc.any(is_function_type))=None,
                           learning_rate:tc.optional(tc.any(numbers.Number, list, np.ndarray, np.matrix))=None,
                           learning_condition:tc.any(Condition, TimeScale,
                                                     tc.enum(UPDATE, CONVERGENCE))=None,
                           context=None):
        """Provide user-accessible-interface to _instantiate_learning_mechanism

        Configure RecurrentTransferMechanism for learning. Creates the following Components:

        * an `AutoAssociativeLearningMechanism` -- if the **learning_function** and/or **learning_rate** arguments are
          specified, they are used to construct the LearningMechanism, otherwise the values specified in the
          RecurrentTransferMechanism's constructor are used;
        ..
        * a `MappingProjection` from the RecurrentTransferMechanism's `primary OutputState <OutputState_Primary>`
          to the AutoAssociativeLearningMechanism's *ACTIVATION_INPUT* InputState;
        ..
        * a `LearningProjection` from the AutoAssociativeLearningMechanism's *LEARNING_SIGNAL* OutputState to
          the RecurrentTransferMechanism's `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

        """

        # This insures that these are validated if the method is called from the command line (i.e., by the user)
        if learning_function:
            self.learning_function = learning_function
        if learning_rate:
            self.learning_rate = learning_rate
        if learning_condition:
            self.learning_condition = learning_condition

        if not isinstance(self.learning_condition, Condition):
            if self.learning_condition is CONVERGENCE:
                self.learning_condition = WhenFinished(self)
            elif self.learning_condition is UPDATE:
                self.learning_condition = None

        self.learning_mechanism = self._instantiate_learning_mechanism(activity_vector=self._learning_signal_source,
                                                                       learning_function=self.learning_function,
                                                                       learning_rate=self.learning_rate,
                                                                       learning_condition=self.learning_condition,
                                                                       matrix=self.recurrent_projection,
                                                                       context=context)

        self.learning_projection = self.learning_mechanism.output_states[LEARNING_SIGNAL].efferents[0]
        if self.learning_mechanism is None:
            self.learning_enabled = False

    def _execute(self, variable=None, context=None, runtime_params=None):

        # if not self.is_initializing
        #     self.parameters.previous_value._set(self.value)
        # self._output = super()._execute(variable=variable, runtime_params=runtime_params, context=context)
        # return self._output
        return super()._execute(variable, context, runtime_params)

    def _parse_function_variable(self, variable, context=None):
        if self.has_recurrent_input_state:
            variable = self.combination_function.execute(variable=variable, context=context)

        return super()._parse_function_variable(variable, context=context)

    def _get_variable_from_input(self, input, context=None):
        if self.has_recurrent_input_state:
            input = np.atleast_2d(input)
            input_len = len(input[0])
            num_inputs = np.size(input, 0)
            num_input_states = len(self.input_states)
            if num_inputs != num_input_states:
                z = np.zeros((1, input_len))
                input = np.concatenate((input, z))

        return super()._get_variable_from_input(input, context)

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, *args, context=None):
        if self.parameters.integrator_mode.get(context):
            super().reinitialize(*args, context=context)
        self.parameters.previous_value.set(None, context, override=True)

    @property
    def _learning_signal_source(self):
        """Return default source of learning signal (`Primary OutputState <OutputState_Primary>)`
              Subclass can override this to provide another source (e.g., see `ContrastiveHebbianMechanism`)
        """
        return self.output_state

    def _get_input_struct_type(self, ctx):
        input_type_list = []
        # FIXME: What if we have more than one state? Does the autoprojection
        # connect only to the first one?
        assert len(self.input_states) == 1
        for state in self.input_states:
            # Extract the non-modulation portion of input state input struct
            s_type = ctx.get_input_struct_type(state).elements[0]
            if isinstance(s_type, pnlvm.ir.ArrayType):
                # Subtract one incoming mapping projections.
                # Unless it's the only incoming projection (mechanism is standalone)
                new_count = max(s_type.count - 1, 1)
                new_type = pnlvm.ir.ArrayType(s_type.element, new_count)
            # FIXME consider struct types
            else:
                assert False
            input_type_list.append(new_type)
        state_input_type_list = []
        for proj in self.mod_afferents:
            state_input_type_list.append(ctx.get_output_struct_type(proj))
        if len(state_input_type_list) > 1:
            input_type_list.append(pnlvm.ir.LiteralStructType(state_input_type_list))
        return pnlvm.ir.LiteralStructType(input_type_list)

    def _get_param_struct_type(self, ctx):
        transfer_t = ctx.get_param_struct_type(super())
        projection_t = ctx.get_param_struct_type(self.recurrent_projection)
        return pnlvm.ir.LiteralStructType([transfer_t, projection_t])

    def _get_state_struct_type(self, ctx):
        transfer_t = ctx.get_state_struct_type(super())
        projection_t = ctx.get_state_struct_type(self.recurrent_projection)
        return_t = ctx.get_output_struct_type(self)
        return pnlvm.ir.LiteralStructType([transfer_t, projection_t, return_t])

    def _get_param_initializer(self, context):
        transfer_params = super()._get_param_initializer(context)
        projection_params = self.recurrent_projection._get_param_initializer(context)
        return tuple([transfer_params, projection_params])

    def _get_state_initializer(self, context):
        transfer_init = super()._get_state_initializer(context)
        projection_init = self.recurrent_projection._get_state_initializer(context)

        # Initialize to output state defaults. That is what the recurrent
        # projection finds.
        retval_init = (tuple(os.defaults.value) if not np.isscalar(os.defaults.value) else os.defaults.value for os in self.output_states)
        return tuple((transfer_init, projection_init, tuple(retval_init)))

    def _gen_llvm_function_body(self, ctx, builder, params, context, arg_in, arg_out):
        real_input_type = super()._get_input_struct_type(ctx)
        real_in = builder.alloca(real_input_type)
        old_val = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(2)])

        # FIXME: What if we have more than one state? Does the autoprojection
        # connect only to the first one?
        assert len(self.input_states) == 1
        for i, state in enumerate(self.input_states):
            is_real_input = builder.gep(real_in, [ctx.int32_ty(0), ctx.int32_ty(i)])
            is_current_input = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(i)])
            for idx in range(len(is_current_input.type.pointee)):
                curr_ptr = builder.gep(is_current_input, [ctx.int32_ty(0), ctx.int32_ty(idx)])
                real_ptr = builder.gep(is_real_input, [ctx.int32_ty(0), ctx.int32_ty(idx)])
                builder.store(builder.load(curr_ptr), real_ptr)

            # FIXME: This is a workaround to find out if we are in a
            #        composition
            if len(state.pathway_projections) == 1:
                continue

            assert len(is_real_input.type.pointee) == len(is_current_input.type.pointee) + 1
            last_idx = len(is_real_input.type.pointee) - 1
            real_last_ptr = builder.gep(is_real_input, [ctx.int32_ty(0), ctx.int32_ty(last_idx)])

            recurrent_f = ctx.get_llvm_function(self.recurrent_projection)
            recurrent_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1)])
            recurrent_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
            # FIXME: Why does this have a wrapper struct?
            recurrent_in = builder.gep(old_val, [ctx.int32_ty(0), ctx.int32_ty(0)])
            builder.call(recurrent_f, [recurrent_params, recurrent_context, recurrent_in, real_last_ptr])

        # Copy mod afferents. These are not impacted by the recurrent projection
        if len(self.mod_afferents) > 1:
            mod_afferent_arg_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(len(self.input_states))])
            mod_afferent_in_ptr = builder.gep(real_in, [ctx.int32_ty(0), ctx.int32_ty(len(self.input_states))])
            builder.store(builder.load(mod_afferent_arg_ptr), mod_afferent_in_ptr)

        transfer_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(0)])
        transfer_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(0)])
        builder = super()._gen_llvm_function_body(ctx, builder, transfer_params, transfer_context, real_in, arg_out)

        builder.store(builder.load(arg_out), old_val)

        return builder

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.recurrent_projection],
        ))
