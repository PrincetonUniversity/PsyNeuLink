# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE PROPRETIES THAT DIRECTLY REFERERENCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ****************************************  RecurrentTransferMechanism *************************************************

"""

Contents
--------

  * `RecurrentTransferMechanism_Overview`
  * `RecurrentTransferMechanism_Creation`
      - `RecurrentTransferMechanism_Learning`
  * `RecurrentTransferMechanism_Structure`
  * `RecurrentTransferMechanism_Execution`
      - `RecurrentTransferMechanism_Execution_Learning`
  * `RecurrentTransferMechanism_Class_Reference`


.. _RecurrentTransferMechanism_Overview:

Overview
--------

A RecurrentTransferMechanism is a subclass of `TransferMechanism` that implements a single-layered recurrent
network, in which each element is connected to every other element (instantiated in a recurrent
`AutoAssociativeProjection` referenced by the Mechanism's `matrix <RecurrentTransferMechanism.matrix>` parameter).
Like a TransferMechanism, it can integrate its input prior to executing its `function <Mechanism_Base.function>`. It
can also report the energy and, if appropriate, the entropy of its output, and can be configured to implement
autoassociative (e.g., Hebbian) learning.

.. _RecurrentTransferMechanism_Creation:

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

.. _RecurrentTransferMechanism_Learning:

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

.. _RecurrentTransferMechanism_Structure:

Structure
---------

The distinguishing feature of a RecurrentTransferMechanism is its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` attribute: a self-projecting `AutoAssociativeProjection`.
By default, `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` projects from the Mechanism's
`primary OutputPort <OutputPort_Primary>` back to its `primary InputPort <InputPort_Primary>`.  This can be
parameterized using its `matrix <RecurrentTransferMechanism.matrix>`, `auto <RecurrentTransferMechanism.auto>`,
and `hetero <RecurrentTransferMechanism.hetero>` attributes, and is stored in its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` attribute.  Using the `has_recurrent_input_port
<RecurrentTransferMechanism.has_recurrent_input_port>` attribute, the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` can also be made to project to a separate *RECURRENT* InputPort
rather, than the primary one (named *EXTERNAL*).  In this case, the InputPorts' results will be combined using the
`combination_function <RecurrentTransferMechanism.combination_function>` *before* being passed to the
RecurrentTransferMechanism's `function <Mechanism_Base.function>`.

A RecurrentTransferMechanism also has two additional `OutputPorts <OutputPort>`:  an *ENERGY* OutputPort and, if its
`function <Mechanism_Base.function>` is bounded between 0 and 1 (e.g., a `Logistic` function), an *ENTROPY* OutputPort.
Each of these report the respective values of the vector in it its *RESULT* (`primary <OutputPort_Primary>`) OutputPort.

Finally, if it has been `specified for learning <RecurrentTransferMechanism_Learning>`, the RecurrentTransferMechanism
is associated with an `AutoAssociativeLearningMechanism` that is used to train its `AutoAssociativeProjection`.
The `learning_enabled <RecurrentTransferMechanism.learning_enabled>` attribute indicates whether learning
is enabled or disabled for the Mechanism.  If learning was not configured when the Mechanism was created, then it cannot
be enabled until the Mechanism is `configured for learning <RecurrentTransferMechanism_Learning>`.

In all other respects the Mechanism is identical to a standard  `TransferMechanism`.

.. _RecurrentTransferMechanism_Execution:

Execution
---------

When a RecurrentTransferMechanism executes, its variable, as is the case with all mechanisms, is determined by the
projections the mechanism receives. This means that a RecurrentTransferMechanism's variable is determined in part by
the value of its own `primary OutputPort <OutputPort_Primary>` on the previous execution, and the `matrix` of the
`recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

Like any `TransferMechanism`, the function used to update each element can be specified in the **function** argument
of its constructor.  This transforms its input (including from the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>`) using the specified function and parameters (see
`TransferMechanism_Execution`), and returns the results in its OutputPorts.  Also like a TransferMechanism,
a RecurrentTransferMechanism can be configured to integrate its input, by setting its `integration_mode
<TransferMechanism.integration_mode>` to True  (see `TransferMechanism_Integration`), and to do so for a
single step of integration or until it reaches some termination condition each time it is executed (see
`TransferMechanism_Termination`). Finally, it can be reset using its `reset
<TransferMechanism.reset>` method (see `TransferMechanism_Reinitialization`).

.. _RecurrentTransferMechanism_Execution_Learning:

*Learning*
~~~~~~~~~~

If the RecurrentTransferMechanism has been `configured for learning <RecurrentTransferMechanism_Learning>` and is
executed as part of a `Composition`, then its `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>`
is executed when the `learning_condition <RecurrentTransferMechanism.learning_condition>` is satisfied.  By default,
the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` executes, and updates the `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` immediately after the RecurrentTransferMechanism executes.

.. _RecurrentTransferMechanism_Class_Reference:

Class Reference
---------------

"""

import itertools
import numbers
import numpy as np
import typecheck as tc
import types
import warnings

from collections.abc import Iterable

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import Function, get_matrix, is_function_type
from psyneulink.core.components.functions.learningfunctions import Hebbian
from psyneulink.core.components.functions.objectivefunctions import Stability
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    ACTIVATION_INPUT, LEARNING_SIGNAL, LearningMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.port import _instantiate_port
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import \
    AUTO, ENERGY, ENTROPY, HETERO, HOLLOW_MATRIX, INPUT_PORT, MATRIX, NAME, RECURRENT_TRANSFER_MECHANISM, RESULT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.registry import register_instance, remove_instance_from_registry
from psyneulink.core.globals.socket import ConnectionInfo
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.core.scheduling.condition import Condition, TimeScale, WhenFinished
from psyneulink.library.components.mechanisms.modulatory.learning.autoassociativelearningmechanism import \
    AutoAssociativeLearningMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import \
    AutoAssociativeProjection, get_auto_matrix, get_hetero_matrix

__all__ = [
    'CONVERGENCE', 'EXTERNAL', 'EXTERNAL_INDEX',
    'RECURRENT', 'RECURRENT_INDEX', 'RecurrentTransferError', 'RecurrentTransferMechanism', 'UPDATE'
]

EXTERNAL = 'EXTERNAL'
RECURRENT = 'RECURRENT'
# Used to index items of InputPort.variable corresponding to recurrent and external inputs
EXTERNAL_INDEX = 0
RECURRENT_INDEX = -1

COMBINATION_FUNCTION = 'combination_function'

# Used to specify learning_condition
UPDATE = 'UPDATE'
CONVERGENCE = 'CONVERGENCE'
ENERGY_OUTPUT_PORT_NAME='ENERGY'
ENTROPY_OUTPUT_PORT_NAME='ENTROPY'



class RecurrentTransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


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
    #     owning_component.recurrent_projection.parameter_ports["matrix"].function.parameters.previous_value._set(value, base_execution_id)

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
    RecurrentTransferMechanism(                             \
        matrix=HOLLOW_MATRIX,                               \
        auto=None,                                          \
        hetero=None,                                        \
        has_recurrent_input_port=False                      \
        combination_function=LinearCombination,             \
        integrator_mode=False,                              \
        integrator_function=AdaptiveIntegrator,             \
        enable_learning=False,                              \
        learning_rate=None,                                 \
        learning_function=Hebbian,                          \
        learning_condition=UPDATE)

    Subclass of `TransferMechanism` that implements a single-layer auto-recurrent network.
    See `TransferMechanism <TransferMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    COMMENT:
    ??OLD OR NEWER THAN BELOW?
    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection
        <RecurrentTransferMechanism_Structure>`, or a AutoAssociativeProjection to use. If **auto** or **hetero**
        arguments are specified, the **matrix** argument will be ignored in favor of those arguments.

    auto : number, 1D array, or None : default None
        specifies matrix as a diagonal matrix with diagonal entries equal to **auto**, if **auto** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. if the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **auto** can be specified as a 1D array with length equal
        to the size of the mechanism, if a non-uniform diagonal is desired. Can be modified by control.

    hetero : number, 2D array, or None : default None
        specifies matrix as a hollow matrix with all non-diagonal entries equal to **hetero**, if **hetero** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. if the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **hetero** can be specified as a 2D array with dimensions
        equal to the matrix dimensions, if a non-uniform diagonal is desired. Can be modified by control.
    COMMENT

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection
        <RecurrentTransferMechanism_Structure>`, or an AutoAssociativeProjection to use.

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

    has_recurrent_input_port : boolean : default False
        specifies whether the mechanism's `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
        points to a separate InputPort. By default, if False, the recurrent_projection points to its `primary
        InputPort <InputPort_Primary>`. If True, the recurrent_projection points to a separate InputPort, and
        the values of all input ports are combined using `LinearCombination <function.LinearCombination>` *before*
        being passed to the RecurrentTransferMechanism's `function <Mechanism_Base.function>`.

    combination_function : function : default LinearCombination
        specifies function used to combine the *RECURRENT* and *INTERNAL* `InputPorts
        <RecurrentTransferMechanism_Structure>`; must accept a 2d array with one or two items of the same length,
        and generate a result that is the same size as each of these;  default simply adds the two items.

    enable_learning : boolean : default False
        specifies whether the Mechanism should be configured for `learning <RecurrentTransferMechanism_Learning>;
        if it is not (the default), then learning cannot be enabled until it is configured for learning by calling
        the Mechanism's `configure_learning <RecurrentTransferMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <RecurrentTransferMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <RecurrentTransferMechanism.learning_rate>` for details).

    learning_function : function : default Hebbian
        specifies the function for the LearningMechanism if `learning has been specified
        <RecurrentTransferMechanism_Learning>` for the RecurrentTransferMechanism.  It can be any function so long as
        it takes a list or 1d array of numeric values as its `variable <Function_Base.variable>` and returns a sqaure
        matrix of numeric values with the same dimensions as the length of the input.

    learning_condition : Condition, UPDATE, CONVERGENCE : default UPDATE
       specifies the `Condition` assigned to `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>`;
       A `Condition` can be used, or one of the following two keywords:

       * *UPDATE:* `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is executed immediately after
         every execution of the RecurrentTransferMechanism;  this is equivalent to assigning no `Condition`
       ..
       * *CONVERGENCE:* `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` is executed whenever the
         the `termination_threshold` is satisfied;  this is equivalent to a WhenFinished(``rec_mech``) `Condition`
         in which ``rec_mech`` is the RecurrentTransferMechanism.

       See `learning_condition <RecurrentTransferMechanism.learning_condition>` for additional details.

    Attributes
    ----------

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection
        <RecurrentTransferMechanism.recurrent_projection>` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputPort <OutputPort_Primary>`
        to its `primary InputPort <Mechanism_InputPorts>`.

    has_recurrent_input_port : boolean
        specifies whether the mechanism's `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
        points to a separate InputPort. If False, the recurrent_projection points to its `primary
        InputPort <InputPort_Primary>`. If True, the recurrent_projection points to a separate InputPort, and
        the values of all input ports are combined using `LinearCombination <function.LinearCombination>` *before*
        being passed to the RecurrentTransferMechanism's `function <Mechanism_Base.function>`.

    combination_function : function
        the Function used to combine the *RECURRENT* and *EXTERNAL* InputPorts if `has_recurrent_input_port
        <RecurrentTransferMechanism.has_recurrent_input_port>` is `True`.  By default this is a `LinearCombination`
        Function that simply adds them.

    learning_enabled : bool
        indicates whether learning has been enabled for the RecurrentTransferMechanism.  It is set to `True` if
        `learning is specified <RecurrentTransferMechanism_Learning>` at the time of construction (i.e., if the
        **enable_learning** argument of the Mechanism's constructor is assigned `True`, or when it is configured for
        learning using the `configure_learning <RecurrentTransferMechanism.configure_learning>` method.  Once learning
        has been configured, `learning_enabled <RecurrentMechahinsm.learning_enabled>` can be toggled at any time to
        enable or disable learning; however, if the Mechanism has not been configured for learning, an attempt to
        set `learning_enabled <RecurrentMechahinsm.learning_enabled>` to `True` elicits a warning and is then
        ignored.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <RecurrentTransferMechanism_Learning>`, and used to train the
        `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the `learning_function <RecurrentTransferMechanism.learning_function>`
        of the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignment).

    learning_function : function
        the function used by the `learning_mechanism <RecurrentTransferMechanism.learning_mechanism>` to train the
        `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` if `learning is specified
        <RecurrentTransferMechanism_Learning>`.

    learning_condition : Condition
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

    standard_output_ports : list[str]
        list of `Standard OutputPorts <OutputPort_Standard>` that includes the following in addition to the
        `standard_output_ports <TransferMechanism.standard_output_ports>` of a `TransferMechanism`:

        *ENERGY* : float
            the energy of the elements in the LCAMechanism's `value <Mechanism_Base.value>`,
            calculated using the `Stability` Function using the `ENERGY` metric.

        .. _LCAMechanism_ENTROPY:

        *ENTROPY* : float
            the entropy of the elements in the LCAMechanism's `value <Mechanism_Base.value>`,
            calculated using the `Stability` Function using the `ENTROPY <CROSS_ENTROPY>` metric.

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
                    :type: ``int``

                combination_function
                    see `combination_function <RecurrentTransferMechanism.combination_function>`

                    :default value: `LinearCombination`
                    :type: `Function`

                enable_learning
                    see `enable_learning <RecurrentTransferMechanism.enable_learning>`

                    :default value: False
                    :type: ``bool``

                has_recurrent_input_port
                    see `has_recurrent_input_port <RecurrentTransferMechanism.has_recurrent_input_port>`

                    :default value: None
                    :type:

                hetero
                    see `hetero <RecurrentTransferMechanism.hetero>`

                    :default value: 0
                    :type: ``int``

                integration_rate
                    see `integration_rate <RecurrentTransferMechanism.integration_rate>`

                    :default value: 0.5
                    :type: ``float``

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
                    :type: ``str``

                noise
                    see `noise <RecurrentTransferMechanism.noise>`

                    :default value: 0.0
                    :type: ``float``

                smoothing_factor
                    see `smoothing_factor <RecurrentTransferMechanism.smoothing_factor>`

                    :default value: 0.5
                    :type: ``float``
        """
        matrix = Parameter(HOLLOW_MATRIX, modulable=True, getter=_recurrent_transfer_mechanism_matrix_getter, setter=_recurrent_transfer_mechanism_matrix_setter)
        auto = Parameter(1, modulable=True)
        hetero = Parameter(0, modulable=True)
        combination_function = LinearCombination
        integration_rate = Parameter(0.5, modulable=True)
        noise = Parameter(0.0, modulable=True)
        smoothing_factor = Parameter(0.5, modulable=True)
        enable_learning = False
        # learning_function is a reference because it is used for
        # an auxiliary learning mechanism
        learning_function = Parameter(
            Hebbian,
            stateful=False,
            loggable=False,
            reference=True
        )
        learning_rate = Parameter(None, setter=_recurrent_transfer_mechanism_learning_rate_setter)
        learning_condition = Parameter(None, stateful=False, loggable=False)
        has_recurrent_input_port = Parameter(False, stateful=False, loggable=False)

        output_ports = Parameter(
            [RESULT],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    standard_output_ports = TransferMechanism.standard_output_ports.copy()
    standard_output_ports.extend([{NAME:ENERGY_OUTPUT_PORT_NAME}, {NAME:ENTROPY_OUTPUT_PORT_NAME}])
    standard_output_port_names = TransferMechanism.standard_output_port_names.copy()
    standard_output_port_names.extend([ENERGY_OUTPUT_PORT_NAME, ENTROPY_OUTPUT_PORT_NAME])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.optional(tc.any(list, dict))) = None,
                 has_recurrent_input_port=None,
                 combination_function: tc.optional(is_function_type) = None,
                 function=None,
                 matrix=None,
                 auto=None,
                 hetero=None,
                 integrator_mode=None,
                 integrator_function=None,
                 initial_value=None,
                 integration_rate: is_numeric_or_none=None,
                 noise=None,
                 clip=None,
                 enable_learning: tc.optional(bool) = None,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.optional(tc.any(is_function_type)) = None,
                 learning_condition:tc.optional(tc.any(Condition, TimeScale,
                                                       tc.enum(UPDATE, CONVERGENCE)))=None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None,
                 **kwargs):
        """Instantiate RecurrentTransferMechanism
        """
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

        super().__init__(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports,
            function=function,
            integrator_function=integrator_function,
            initial_value=initial_value,
            noise=noise,
            matrix=matrix,
            integrator_mode=integrator_mode,
            integration_rate=integration_rate,
            learning_rate=learning_rate,
            learning_function=learning_function,
            learning_condition=learning_condition,
            auto=auto,
            hetero=hetero,
            has_recurrent_input_port=has_recurrent_input_port,
            combination_function=combination_function,
            clip=clip,
            output_ports=output_ports,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

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
                    raise RecurrentTransferError("{0} has non-numeric entries".format(matrix))
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
        ParameterPorts for auto and hetero if they haven't already been instantiated. This is useful if auto and
        hetero were None in the initialization call.
        :param function:
        """
        super()._instantiate_attributes_before_function(function=function, context=context)

        param_keys = self._parameter_ports.key_values

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
                                         "auto or hetero parameter is missing.".format(self.matrix, self))

        if AUTO not in param_keys and HETERO in param_keys:
            d = np.diagonal(matrix).copy()
            port = _instantiate_port(owner=self,
                                       port_type=ParameterPort,
                                       name=AUTO,
                                       reference_value=d,
                                       reference_value_name=AUTO,
                                       params=None,
                                       context=context)
            self.auto = d
            if port is not None:
                self._parameter_ports[AUTO] = port
                port.source = self
            else:
                raise RecurrentTransferError("Failed to create ParameterPort for `auto` attribute for {} \"{}\"".
                                           format(self.__class__.__name__, self.name))
        if HETERO not in param_keys and AUTO in param_keys:

            m = matrix.copy()
            np.fill_diagonal(m, 0.0)
            self.hetero = m
            port = _instantiate_port(owner=self,
                                       port_type=ParameterPort,
                                       name=HETERO,
                                       reference_value=m,
                                       reference_value_name=HETERO,
                                       params=None,
                                       context=context)
            if port is not None:
                self._parameter_ports[HETERO] = port
                port.source = self
            else:
                raise RecurrentTransferError("Failed to create ParameterPort for `hetero` attribute for {} \"{}\"".
                                           format(self.__class__.__name__, self.name))

        if self.has_recurrent_input_port:
            comb_fct = self.combination_function
            if (
                not (
                    isinstance(comb_fct, LinearCombination)
                    or (isinstance(comb_fct, type) and issubclass(comb_fct, LinearCombination))
                    or (isinstance(comb_fct, types.MethodType) and comb_fct.__self__ == self)
                )
            ):
                if isinstance(comb_fct, type):
                    comb_fct = comb_fct()
                elif isinstance(comb_fct, (types.FunctionType, types.MethodType)):
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
                elif isinstance(comb_fct, types.MethodType) and comb_fct.__self__ == self:
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
        """Instantiate recurrent_projection, matrix, and the functions for the ENERGY and ENTROPY OutputPorts
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

            # creating a recurrent_projection changes the default variable shape
            # so we have to reshape any Paramter Functions
            self._update_parameter_class_variables(context)

        self.aux_components.append(self.recurrent_projection)

        if self.learning_enabled:
            self.configure_learning(context=context)

        if ENERGY_OUTPUT_PORT_NAME in self.output_ports.names:
            energy = Stability(self.defaults.variable[0],
                               metric=ENERGY,
                               transfer_fct=self.function,
                               matrix=self.recurrent_projection._parameter_ports[MATRIX])
            self.output_ports[ENERGY_OUTPUT_PORT_NAME]._calculate = energy.function

        if ENTROPY_OUTPUT_PORT_NAME in self.output_ports.names:
            if self.function.bounds == (0,1) or self.clip == (0,1):
                entropy = Stability(self.defaults.variable[0],
                                    metric=ENTROPY,
                                    transfer_fct=self.function,
                                    matrix=self.recurrent_projection._parameter_ports[MATRIX])
                self.output_ports[ENTROPY_OUTPUT_PORT_NAME]._calculate = entropy.function
            else:
                del self.output_ports[ENTROPY_OUTPUT_PORT_NAME]

    def _update_parameter_ports(self, runtime_params=None, context=None):
        for port in self._parameter_ports:
            # (8/2/17 CW) because the auto and hetero params are solely used by the AutoAssociativeProjection
            # (the RecurrentTransferMechanism doesn't use them), the auto and hetero param ports are updated in the
            # projection's _update_parameter_ports, and accordingly are not updated here
            if port.name != AUTO and port.name != HETERO:
                port._update(params=runtime_params, context=context)

    @property
    def recurrent_size(self):
        return len(self.defaults.variable[0])

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
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = val

        self.parameters.matrix._set(val, self.most_recent_context)

        if hasattr(self, '_parameter_ports') and 'matrix' in self._parameter_ports:
            param_port = self._parameter_ports['matrix']

            if hasattr(param_port.function, 'initializer'):
                param_port.function.reset = val

    @property
    def auto(self):
        return self.parameters.auto._get(self.most_recent_context)

    @auto.setter
    def auto(self, val):
        self.parameters.auto._set(val, self.most_recent_context)

        if hasattr(self, "recurrent_projection") and 'hetero' in self._parameter_ports:
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = self.matrix


    @property
    def hetero(self):
        return self.parameters.hetero._get(self.most_recent_context)

    @hetero.setter
    def hetero(self, val):
        self.parameters.hetero._set(val, self.most_recent_context)

        if hasattr(self, "recurrent_projection") and 'auto' in self._parameter_ports:
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = self.matrix_param

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
        if self.has_recurrent_input_port:
            # # FIX: 7/12/18 MAKE THIS A METHOD THAT CAN BE OVERRIDDEN BY CONTRASTIVEHEBBIAN
            new_input_port = InputPort(owner=self, name=RECURRENT, variable=self.defaults.variable[0],
                                        internal_only=True)
            assert (len(new_input_port.all_afferents) == 0)  # just a sanity check
            assert(self.input_port.name != "Recurrent Input Port")
            # Rename existing InputPort as EXTERNAL
            remove_instance_from_registry(registry=self._portRegistry,
                                          category=INPUT_PORT,
                                          component=self.input_port)
            register_instance(self.input_port, EXTERNAL, InputPort, self._portRegistry, INPUT_PORT)
            proj = AutoAssociativeProjection(owner=mech,
                                             receiver=new_input_port,
                                             matrix=matrix,
                                             name=mech.name + ' recurrent projection')
            receiver = new_input_port
        else:
            proj = AutoAssociativeProjection(owner=mech,
                                             matrix=matrix,
                                             name=mech.name + ' recurrent projection')
            receiver = self.input_port

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

        from psyneulink.core.compositions.composition import NodeRole

        learning_mechanism.condition = learning_condition
        # # MODIFIED 10/23/19 OLD:
        # self.aux_components.append(learning_mechanism)
        # MODIFIED 10/23/19 NEW:
        self.aux_components.append((learning_mechanism, NodeRole.LEARNING))
        # MODIFIED 10/23/19 END
        # Instantiate Projection from Mechanism's output to LearningMechanism
        mproj = MappingProjection(sender=activity_vector,
                          receiver=learning_mechanism.input_ports[ACTIVATION_INPUT],
                          name="Error Projection for {}".format(learning_mechanism.name))
        mproj._activate_for_all_compositions()
        self.aux_components.append(mproj)
        # Instantiate Projection from LearningMechanism to Mechanism's AutoAssociativeProjection
        lproj = LearningProjection(sender=learning_mechanism.output_ports[LEARNING_SIGNAL],
                           receiver=matrix.parameter_ports[MATRIX],
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
        * a `MappingProjection` from the RecurrentTransferMechanism's `primary OutputPort <OutputPort_Primary>`
          to the AutoAssociativeLearningMechanism's *ACTIVATION_INPUT* InputPort;
        ..
        * a `LearningProjection` from the AutoAssociativeLearningMechanism's *LEARNING_SIGNAL* OutputPort to
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
            if self.learning_condition == CONVERGENCE:
                self.learning_condition = WhenFinished(self)
            elif self.learning_condition == UPDATE:
                self.learning_condition = None

        self.learning_mechanism = self._instantiate_learning_mechanism(activity_vector=self._learning_signal_source,
                                                                       learning_function=self.learning_function,
                                                                       learning_rate=self.learning_rate,
                                                                       learning_condition=self.learning_condition,
                                                                       matrix=self.recurrent_projection,
                                                                       context=context)

        self.learning_projection = self.learning_mechanism.output_ports[LEARNING_SIGNAL].efferents[0]
        if self.learning_mechanism is None:
            self.learning_enabled = False

    def _execute(self, variable=None, context=None, runtime_params=None):

        # if not self.is_initializing
        #     self.parameters.previous_value._set(self.value)
        # self._output = super()._execute(variable=variable, runtime_params=runtime_params, context=context)
        # return self._output
        return super()._execute(variable, context, runtime_params)

    def _parse_function_variable(self, variable, context=None):
        if self.has_recurrent_input_port:
            variable = self.combination_function.execute(variable=variable, context=context)

        return super()._parse_function_variable(variable, context=context)

    def _get_variable_from_input(self, input, context=None):
        if self.has_recurrent_input_port:
            input = np.atleast_2d(input)
            input_len = len(input[0])
            num_inputs = np.size(input, 0)
            num_input_ports = len(self.input_ports)
            if num_inputs != num_input_ports:
                z = np.zeros((1, input_len))
                input = np.concatenate((input, z))

        return super()._get_variable_from_input(input, context)

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, force=False, context=None):
        super().reset(*args, force=force, context=context)
        self.parameters.value.clear_history(context)

    @property
    def _learning_signal_source(self):
        """Return default source of learning signal (`Primary OutputPort <OutputPort_Primary>)`
              Subclass can override this to provide another source (e.g., see `ContrastiveHebbianMechanism`)
        """
        return self.output_port

    def _get_param_ids(self):
        return super()._get_param_ids() + ["recurrent_projection"]

    def _get_param_struct_type(self, ctx):
        transfer_t = ctx.get_param_struct_type(super())
        projection_t = ctx.get_param_struct_type(self.recurrent_projection)
        return pnlvm.ir.LiteralStructType([*transfer_t.elements, projection_t])

    def _get_state_ids(self):
        return super()._get_state_ids() + ["old_val", "recurrent_projection"]

    def _get_state_struct_type(self, ctx):
        transfer_t = ctx.get_state_struct_type(super())
        projection_t = ctx.get_state_struct_type(self.recurrent_projection)
        return_t = ctx.get_output_struct_type(self)
        return pnlvm.ir.LiteralStructType([*transfer_t.elements, return_t, projection_t])

    def _get_param_initializer(self, context):
        transfer_params = super()._get_param_initializer(context)
        projection_params = self.recurrent_projection._get_param_initializer(context)
        return (*transfer_params, projection_params)

    def _get_state_initializer(self, context):
        transfer_init = super()._get_state_initializer(context)
        projection_init = self.recurrent_projection._get_state_initializer(context)

        # Initialize to OutputPort defaults.
        # That is what the recurrent projection finds.
        retval_init = (tuple(op.parameters.value.get(context)) if not np.isscalar(op.parameters.value.get(context)) else op.parameters.value.get(context) for op in self.output_ports)
        return (*transfer_init, tuple(retval_init), projection_init)

    def _gen_llvm_function_reset(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" in tags

        # Check if we have reinitializers
        has_reinitializers_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, "has_initializers")
        has_initializers = builder.load(has_reinitializers_ptr)
        not_initializers = builder.fcmp_ordered("==", has_initializers,
                                                has_initializers.type(0))
        with builder.if_then(not_initializers):
            builder.ret_void()

        # Reinit main function. This is a no-op if it's not a stateful function.
        reinit_func = ctx.import_llvm_function(self.function, tags=tags)
        reinit_params = pnlvm.helpers.get_param_ptr(builder, self, params, "function")
        reinit_state = pnlvm.helpers.get_state_ptr(builder, self, state, "function")
        reinit_in = builder.alloca(reinit_func.args[2].type.pointee)
        reinit_out = builder.alloca(reinit_func.args[3].type.pointee)
        builder.call(reinit_func, [reinit_params, reinit_state, reinit_in,
                                   reinit_out])

        # Reinit integrator function
        if self.integrator_mode:
            reinit_f = ctx.import_llvm_function(self.integrator_function,
                                                tags=tags)
            reinit_in = builder.alloca(reinit_f.args[2].type.pointee)
            reinit_out = builder.alloca(reinit_f.args[3].type.pointee)
            reinit_params = pnlvm.helpers.get_param_ptr(builder, self, params, "integrator_function")
            reinit_state = pnlvm.helpers.get_state_ptr(builder, self, state, "integrator_function")
            builder.call(reinit_f, [reinit_params, reinit_state, reinit_in,
                                    reinit_out])

        prev_val_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "old_val")
        builder.store(prev_val_ptr.type.pointee(None), prev_val_ptr)
        return builder

    def _gen_llvm_input_ports(self, ctx, builder, params, state, arg_in):
        recurrent_state = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                      "recurrent_projection")
        recurrent_params = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                       "recurrent_projection")
        recurrent_f = ctx.import_llvm_function(self.recurrent_projection)

        # Extract the correct output port value
        prev_val_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "old_val")
        recurrent_in = builder.gep(prev_val_ptr, [ctx.int32_ty(0),
            ctx.int32_ty(self.output_ports.index(self.recurrent_projection.sender))])

        # Get the correct recurrent output location
        recurrent_out = builder.gep(arg_in, [ctx.int32_ty(0),
                                             ctx.int32_ty(self.input_ports.index(self.recurrent_projection.receiver)),
                                             ctx.int32_ty(self.recurrent_projection.receiver.pathway_projections.index(self.recurrent_projection))])

        # the recurrent projection is not executed in standalone mode
        if len(self.afferents) == 1:
            assert self.path_afferents[0] is self.recurrent_projection
            # NOTE: we should zero the target location here, but in standalone
            # mode ctypes does it for us when instantiating the input structure
        else:
            # Execute the recurrent projection here. This makes it part of the
            # 'is_finished' inner loop so we always see the most up-to-date
            # input
            builder.call(recurrent_f, [recurrent_params, recurrent_state, recurrent_in, recurrent_out])

        return super()._gen_llvm_input_ports(ctx, builder, params, state, arg_in)

    def _gen_llvm_output_ports(self, ctx, builder, value,
                               mech_params, mech_state, mech_in, mech_out):
        ret = super()._gen_llvm_output_ports(ctx, builder, value, mech_params,
                                             mech_state, mech_in, mech_out)

        old_val = pnlvm.helpers.get_state_ptr(builder, self, mech_state, "old_val")
        builder.store(builder.load(mech_out), old_val)
        return ret

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.recurrent_projection],
        ))
