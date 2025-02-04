# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  GRUMechanism *************************************************

"""

Contents
--------

  * `GRUMechanism_Overview`
  * `GRUMechanism_Creation`
      - `GRUMechanism_Learning`
  * `GRUMechanism_Structure`
  * `GRUMechanism_Execution`
      - `GRUMechanism_Execution_Learning`
  * `GRUMechanism_Class_Reference`


.. _GRUMechanism_Overview:

Overview
--------

A GRUMechanism is a subclass of `RecurrentTransferMechanism` that implements a single-layered gated recurrent
network, in which each element is connected to every other element (instantiated in a recurrent
`AutoAssociativeProjection` referenced by the Mechanism's `matrix <GRUMechanism.matrix>` parameter). The primary
difference between a GRUMechanism and a standard `RecurrentTransferMechanism` is that the GRUMechanism has two
additional `InputPorts <InputPort>`: a *RESET* InputPort and *UPDATE* InputPort. These receive Projections from the
same source as the Mechanism's `primary InputPort <InputPort_Primary>`, and are combined with that input before being
passed to its `function <GRUMechanism.function>` in order to modulate the flow of information through the Mechanism.

.. _GRUMechanism_Creation:

Creating a GRUMechanism
-------------------------------------

A GRUMechanism is created directly by calling its constructor, for example::

    import psyneulink as pnl
    my_linear_recurrent_transfer_mechanism = pnl.GRUMechanism(function=pnl.Linear)
    my_logistic_recurrent_transfer_mechanism = pnl.GRUMechanism(function=pnl.Logistic(gain=1.0,
                                                                                                    bias=-4.0))

The recurrent projection is automatically created using (1) the **matrix** argument or (2) the **auto** and **hetero**
arguments of the Mechanism's constructor, and is assigned to the mechanism's `recurrent_projection
<GRUMechanism.recurrent_projection>` attribute.

If the **matrix** argument is used to create the `recurrent_projection
<GRUMechanism.recurrent_projection>`, it must specify either a square matrix or an
`AutoAssociativeProjection` that uses one (the default is `HOLLOW_MATRIX`).::

    recurrent_mech_1 = pnl.GRUMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=[[1.0, 2.0, 2.0],
                                                              [2.0, 1.0, 2.0],
                                                              [2.0, 2.0, 1.0]])

    recurrent_mech_2 = pnl.GRUMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      matrix=pnl.AutoAssociativeProjection)

If the **auto** and **hetero** arguments are used to create the `recurrent_projection
<GRUMechanism.recurrent_projection>`, they set the diagonal and off-diagonal terms, respectively.::

    recurrent_mech_3 = pnl.GRUMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                      auto=1.0,
                                                      hetero=2.0)

.. note::

    In the examples above, recurrent_mech_1 and recurrent_mech_3 are identical.

In all other respects, a GRUMechanism is specified in the same way as a standard `TransferMechanism`.

.. _GRUMechanism_Learning:

*Configuring Learning*
~~~~~~~~~~~~~~~~~~~~~~

A GRUMechanism can be configured for learning when it is created by assigning `True` to the
**enable_learning** argument of its constructor.  This creates an `AutoAssociativeLearningMechanism` that is used to
train its `recurrent_projection <GRUMechanism.recurrent_projection>`, and assigns as its `function
<Function_Base.function>` the one  specified in the **learning_function** argument of the GRUMechanism's
constructor.  By default, this is the `Hebbian` Function;  however, it can be replaced by any other function that is
suitable for autoassociative learning;  that is, one that takes a list or 1d array of numeric values
(an "activity vector") and returns a 2d array or square matrix (the "weight change matrix") with the same dimensions
as the length of the activity vector. The AutoAssociativeLearningMechanism is assigned to the `learning_mechanism
<GRUMechanism.learning_mechanism>` attribute and is used to modify the `matrix
<AutoAssociativeProjection.matrix>` parameter of its `recurrent_projection
<GRUMechanism.recurrent_projection>` (also referenced by the GRUMechanism's own `matrix
<GRUMechanism.matrix>` parameter.

If a GRUMechanism is created without configuring learning (i.e., **enable_learning** is assigned `False`
in its constructor -- the default value), then learning cannot be enabled for the Mechanism until it has been
configured for learning;  any attempt to do so will issue a warning and then be ignored.  Learning can be configured
once the Mechanism has been created by calling its `configure_learning <GRUMechanism.configure_learning>`
method, which also enables learning.

COMMENT:
8/7/17 CW: In past versions, the first sentence of the paragraph above was: "A GRUMechanism can be
created directly by calling its constructor, or using the `mechanism() <Mechanism.mechanism>` command and specifying
RECURRENT_TRANSFER_MECHANISM as its **mech_spec** argument".
However, the latter method is no longer correct: it instead creates a DDM: the problem is line 590 in Mechanism.py,
as MechanismRegistry is empty!
10/9/17 MANTEL: mechanism() factory method is removed
COMMENT

.. _GRUMechanism_Structure:

Structure
---------

The distinguishing feature of a GRUMechanism is its `recurrent_projection
<GRUMechanism.recurrent_projection>` attribute: a self-projecting `AutoAssociativeProjection`.
By default, `recurrent_projection <GRUMechanism.recurrent_projection>` projects from the Mechanism's
`primary OutputPort <OutputPort_Primary>` back to its `primary InputPort <InputPort_Primary>`.  This can be
parameterized using its `matrix <GRUMechanism.matrix>`, `auto <GRUMechanism.auto>`,
and `hetero <GRUMechanism.hetero>` attributes, and is stored in its `recurrent_projection
<GRUMechanism.recurrent_projection>` attribute.  Using the `has_recurrent_input_port
<GRUMechanism.has_recurrent_input_port>` attribute, the `recurrent_projection
<GRUMechanism.recurrent_projection>` can also be made to project to a separate *RECURRENT* InputPort
rather, than the primary one (named *EXTERNAL*).  In this case, the InputPorts' results will be combined using the
`combination_function <GRUMechanism.combination_function>` *before* being passed to the
GRUMechanism's `function <Mechanism_Base.function>`.

A GRUMechanism also has two additional `OutputPorts <OutputPort>`:  an *ENERGY* OutputPort and, if its
`function <Mechanism_Base.function>` is bounded between 0 and 1 (e.g., a `Logistic` function), an *ENTROPY* OutputPort.
Each of these report the respective values of the vector in it its *RESULT* (`primary <OutputPort_Primary>`) OutputPort.

Finally, if it has been `specified for learning <GRUMechanism_Learning>`, the GRUMechanism
is associated with an `AutoAssociativeLearningMechanism` that is used to train its `AutoAssociativeProjection`.
The `learning_enabled <GRUMechanism.learning_enabled>` attribute indicates whether learning
is enabled or disabled for the Mechanism.  If learning was not configured when the Mechanism was created, then it cannot
be enabled until the Mechanism is `configured for learning <GRUMechanism_Learning>`.

In all other respects the Mechanism is identical to a standard  `TransferMechanism`.

.. _GRUMechanism_Execution:

Execution
---------

When a GRUMechanism executes, its variable, as is the case with all mechanisms, is determined by the
projections the mechanism receives. This means that a GRUMechanism's variable is determined in part by
the value of its own `primary OutputPort <OutputPort_Primary>` on the previous execution, and the `matrix` of the
`recurrent_projection <GRUMechanism.recurrent_projection>`.

Like any `TransferMechanism`, the function used to update each element can be specified in the **function** argument
of its constructor.  This transforms its input (including from the `recurrent_projection
<GRUMechanism.recurrent_projection>`) using the specified function and parameters (see
`TransferMechanism_Execution`), and returns the results in its OutputPorts.  Also like a TransferMechanism,
a GRUMechanism can be configured to integrate its input, by setting its `integration_mode
<TransferMechanism.integration_mode>` to True  (see `TransferMechanism_Execution_With_Integration`), and to do so for a
single step of integration or until it reaches some termination condition each time it is executed (see
`TransferMechanism_Execution_Integration_Termination`). Finally, it can be reset using its `reset
<TransferMechanism.reset>` method (see `TransferMechanism_Execution_Integration_Reinitialization`).

.. _GRUMechanism_Execution_Learning:

*Learning*
~~~~~~~~~~

If the GRUMechanism has been `configured for learning <GRUMechanism_Learning>` and is
executed as part of a `Composition`, then its `learning_mechanism <GRUMechanism.learning_mechanism>`
is executed when the `learning_condition <GRUMechanism.learning_condition>` is satisfied.  By default,
the `learning_mechanism <GRUMechanism.learning_mechanism>` executes, and updates the `recurrent_projection
<GRUMechanism.recurrent_projection>` immediately after the GRUMechanism executes.

.. _GRUMechanism_Class_Reference:

Class Reference
---------------

"""

import copy
import numbers
import types
import warnings
from collections.abc import Iterable

import numpy as np
from beartype import beartype

from psyneulink._typing import Optional, Union, Callable, Literal

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import _get_parametervalue_attr
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear
from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination
from psyneulink.core.components.functions.function import Function, get_matrix
from psyneulink.core.components.functions.nonstateful.learningfunctions import Hebbian
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Stability, Energy, Entropy
from psyneulink.core.components.functions.stateful.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base, MechanismError
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    ACTIVATION_INPUT, LEARNING_SIGNAL, LearningMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.port import _instantiate_port
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import \
    (AUTO, ENERGY, ENTROPY, FUNCTION, HETERO, HOLLOW_MATRIX, INPUT_PORT,
     MATRIX, NAME, RECURRENT_TRANSFER_MECHANISM, RESULT)
from psyneulink.core.globals.parameters import Parameter, SharedParameter, check_user_specified, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.registry import register_instance, remove_instance_from_registry
from psyneulink.core.globals.socket import ConnectionInfo
from psyneulink.core.globals.utilities import NumericCollections, ValidParamSpecType, safe_len
from psyneulink.core.scheduling.condition import Condition, WhenFinished
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.modulatory.learning.autoassociativelearningmechanism import \
    AutoAssociativeLearningMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import \
    AutoAssociativeProjection, get_auto_matrix, get_hetero_matrix
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferMechanism, _recurrent_transfer_mechanism_matrix_getter, _recurrent_transfer_mechanism_matrix_setter

__all__ = [
    'CONVERGENCE', 'EXTERNAL', 'EXTERNAL_INDEX',
    'RECURRENT', 'RECURRENT_INDEX', 'RecurrentTransferError', 'GRUMechanism', 'UPDATE'
]

EXTERNAL = 'EXTERNAL'
RECURRENT = 'RECURRENT'
# Used to index items of InputPort.variable corresponding to recurrent and external inputs
EXTERNAL_INDEX = 0
RECURRENT_INDEX = -1
RESET_INDEX = 1
RESET_UPDATE = 2
RESET_NEW = 2


COMBINATION_FUNCTION = 'combination_function'

# Used to specify learning_condition
UPDATE = 'UPDATE'
CONVERGENCE = 'CONVERGENCE'
ENERGY_OUTPUT_PORT_NAME=ENERGY
ENTROPY_OUTPUT_PORT_NAME=ENTROPY


class RecurrentTransferError(MechanismError):
    pass


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
    # if owning_component.recurrent_projection is not None:
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


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class GRUMechanism(RecurrentTransferMechanism):
    """
    GRUMechanism(             \
        reset_weights=None,   \
        update_weights=None,  \
        new_weights=None,     \

    Subclass of `TransferMechanism` that implements a single-layer gated recurrent network.
    See `RecurrentTransferMechanism <RecurrentTransferMechanism_Class_Reference>` for additional args and attributes.

    Arguments
    ---------

    reset_weights : 2d np.array : default RANDOM_CONNECTIVITY_MATRIX
        specifies the matrix for the weights from the input to the reset_gate.

    update_weights : 2d np.array : default RANDOM_CONNECTIVITY_MATRIX
        specifies the matrix for the weights from the input to the update_gate.

    new_weights : 2d np.array : default RANDOM_CONNECTIVITY_MATRIX
        specifies the matrix for the weights from the input to the new_gate.

    Attributes
    ----------

    reset_gate : 1d np.array
        value of the reset_gate.

    update_gate : 1d np.array
        value of the update_gate.

    new_gate : 1d np.array
        value of the new_gate.

    reset_weights : 2d np.array
        weight matrix from the input to the reset_gate.

    update_weights : 2d np.array
        weight matrix from the input to the update_gate.

    new_weights : 2d np.array
        weight matrix from the input to the new_gate.


    Returns
    -------
    instance of GRUMechanism : GRUMechanism

    """
    componentType = GRU_MECHANISM

    class Parameters(RecurrentTransferMechanism.Parameters):
        """
            Attributes
            ----------

                reset_gate_weights
                    see `matrix <GRUMechanism.matrix>`

                    :default value: `HOLLOW_MATRIX`
                    :type: ``str``

        """
        reset_gate = Parameter(None, stateful=False, loggable=False, structural=True)
        update_gate = Parameter(None, stateful=False, loggable=False, structural=True)
        new_gate = Parameter(None, stateful=False, loggable=False, structural=True)
        reset_weights = Parameter(None, stateful=False, loggable=False, structural=True)
        update_weights = Parameter(None, stateful=False, loggable=False, structural=True)
        new_weights = Parameter(None, stateful=False, loggable=False, structural=True)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 input_ports: Optional[Union[list, dict]] = None,
                 has_recurrent_input_port=None,
                 combination_function: Optional[Callable] = None,
                 function=None,
                 matrix=None,
                 auto=None,
                 hetero=None,
                 integrator_mode=None,
                 integrator_function=None,
                 initial_value=None,
                 integration_rate: Optional[NumericCollections] = None,
                 noise=None,
                 clip=None,
                 enable_learning: Optional[bool] = None,
                 learning_rate: Optional[Union[ValidParamSpecType, bool]] = None,
                 learning_function: Optional[Callable] = None,
                 learning_condition: Optional[Union[Condition, TimeScale, Literal['UPDATE', 'CONVERGENCE']]] = None,
                 output_ports: Optional[Union[str, Iterable]] = None,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 **kwargs):
        """Instantiate GRUMechanism
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

        self.learning_mechanism = None
        self._learning_enabled = enable_learning

        super().__init__(
            default_variable=default_variable,
            input_shapes=input_shapes,
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
                               format(MATRIX, self.name, rows, self.__class__.__name__, self.name, self.input_shapes,
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
            if isinstance(auto_param, (np.ndarray, list)) and safe_len(auto_param) != 1 and safe_len(auto_param) != self.input_shapes[0]:
                raise RecurrentTransferError("auto parameter ({0}) for {1} is of incompatible length with the size "
                                             "({2}) of its owner, {1}.".format(auto_param, self, self.input_shapes[0]))

        if HETERO in target_set:
            hetero_param = target_set[HETERO]
            if hetero_param is not None and not isinstance(hetero_param, (np.ndarray, list, numbers.Number)):
                raise RecurrentTransferError("hetero parameter ({}) of {} is of incompatible type: it should be a "
                                             "number, None, or a 2D numeric array".format(hetero_param, self))
            hetero_shape = np.array(hetero_param).shape
            if hetero_shape != (1,) and hetero_shape != (1, 1):
                if isinstance(hetero_param, (np.ndarray, list, np.matrix)) and (hetero_param.ndim > 0 and hetero_shape[0] != self.input_shapes[0]):
                    raise RecurrentTransferError("hetero parameter ({0}) for {1} is of incompatible size with the size "
                                                 "({2}) of its owner, {1}.".format(hetero_param, self, self.input_shapes[0]))
                if isinstance(hetero_param, (np.ndarray, list, np.matrix)) and (hetero_param.ndim > 0 and hetero_shape[0] != hetero_shape[1]):
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

        matrix = get_matrix(copy_parameter_value(self.defaults.matrix), rows=self.recurrent_size, cols=self.recurrent_size)

        # below implements the rules provided by KAM:
        # - If auto and hetero but not matrix are specified, the diagonal terms of the matrix are determined by auto and the off-diagonal terms are determined by hetero.
        # - If auto, hetero, and matrix are all specified, matrix is ignored in favor of auto and hetero.
        # - If auto and matrix are both specified, the diagonal terms are determined by auto and the off-diagonal terms are determined by matrix. ​
        # - If hetero and matrix are both specified, the diagonal terms are determined by matrix and the off-diagonal terms are determined by hetero.
        auto = get_auto_matrix(copy_parameter_value(self.defaults.auto), self.recurrent_size)
        hetero = get_hetero_matrix(copy_parameter_value(self.defaults.hetero), self.recurrent_size)
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
                                         "auto or hetero parameter is missing.".format(self.parameters.matrix._get(context), self))

        if AUTO not in param_keys and HETERO in param_keys:
            d = np.diagonal(matrix).copy()
            port = _instantiate_port(owner=self,
                                       port_type=ParameterPort,
                                       name=AUTO,
                                       reference_value=d,
                                       reference_value_name=AUTO,
                                       params=None,
                                       context=context)
            self.parameters.auto._set(d, context)
            if port is not None:
                self._parameter_ports[AUTO] = port
                port.source = self.parameters.auto
            else:
                raise RecurrentTransferError("Failed to create ParameterPort for `auto` attribute for {} \"{}\"".
                                           format(self.__class__.__name__, self.name))
        if HETERO not in param_keys and AUTO in param_keys:

            m = matrix.copy()
            np.fill_diagonal(m, 0.0)
            self.parameters.hetero._set(m, context)
            port = _instantiate_port(owner=self,
                                       port_type=ParameterPort,
                                       name=HETERO,
                                       reference_value=m,
                                       reference_value_name=HETERO,
                                       params=None,
                                       context=context)
            if port is not None:
                self._parameter_ports[HETERO] = port
                port.source = self.parameters.hetero
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

        if self.parameters.auto._get(context) is None and self.parameters.hetero._get(context) is None:
            self.parameters.matrix._set(matrix, context)
            if self.parameters.matrix._get(context) is None:
                raise RecurrentTransferError("PROGRAM ERROR: Failed to instantiate \'matrix\' param for {}".
                                             format(self.__class__.__name__))

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate recurrent_projection, matrix, and the functions for the ENERGY and ENTROPY OutputPorts
        """
        from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

        matrix = self.parameters.matrix._get(context)

        # Now that matrix and default_variable size are known,
        #     instantiate functions for ENERGY and ENTROPY standard_output_ports
        if ENERGY_OUTPUT_PORT_NAME in self.output_ports:
            energy_idx = self.standard_output_port_names.index(ENERGY_OUTPUT_PORT_NAME)
            self.standard_output_ports[energy_idx][FUNCTION] = Energy(self.defaults.variable,
                                                                      matrix=matrix)
        if ENTROPY_OUTPUT_PORT_NAME in self.output_ports:
            energy_idx = self.standard_output_port_names.index(ENTROPY_OUTPUT_PORT_NAME)
            self.standard_output_ports[energy_idx][FUNCTION] = Entropy(self.defaults.variable)

        super()._instantiate_attributes_after_function(context=context)

        # (7/19/17 CW) this line of code is now questionable, given the changes to matrix and the recurrent projection
        if isinstance(matrix, AutoAssociativeProjection):
            self.recurrent_projection = matrix

        # IMPLEMENTATION NOTE:  THESE SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
        else:
            self.recurrent_projection = self._instantiate_recurrent_projection(self,
                                                                               matrix=matrix,
                                                                               context=context)

            # creating a recurrent_projection changes the default variable shape
            # so we have to reshape any Paramter Functions
            self._update_default_variable(self.defaults.variable, context)

        self.aux_components.append(self.recurrent_projection)

        if self.learning_enabled:
            self.configure_learning(context=context)

    def _update_parameter_ports(self, runtime_params=None, context=None):
        for port in self._parameter_ports:
            # (8/2/17 CW) because the auto and hetero params are solely used by the AutoAssociativeProjection
            # (the GRUMechanism doesn't use them), the auto and hetero param ports are updated in the
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
        return getattr(self, _get_parametervalue_attr(self.parameters.matrix))

    @matrix.setter
    def matrix(self, val): # simplified version of standard setter (in Component.py)
        # KDM 10/12/18: removing below because it doesn't seem to be correct, and also causes
        # unexpected values to be set to previous_value
        # KDM 7/1/19: reinstating below
        if self.recurrent_projection is not None:
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = val
            self.recurrent_projection.parameter_ports["matrix"].function.reset = val

        self.parameters.matrix.set(val, self.most_recent_context)

    @property
    def auto(self):
        return getattr(self, _get_parametervalue_attr(self.parameters.auto))

    @auto.setter
    def auto(self, val):
        self.parameters.auto.set(val, self.most_recent_context)

        if self.recurrent_projection is not None and 'hetero' in self._parameter_ports:
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = self.matrix

    @property
    def hetero(self):
        return getattr(self, _get_parametervalue_attr(self.parameters.hetero))

    @hetero.setter
    def hetero(self, val):
        self.parameters.hetero.set(val, self.most_recent_context)

        if self.recurrent_projection is not None and 'auto' in self._parameter_ports:
            self.recurrent_projection.parameter_ports["matrix"].function.previous_value = self.matrix_param

    @property
    def learning_enabled(self):
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value:bool):

        self._learning_enabled = value
        # Enable learning for GRUMechanism's learning_mechanism
        if self.learning_mechanism is not None:
            self.learning_mechanism.learning_enabled = value
        # If GRUMechanism has no LearningMechanism, warn and then ignore attempt to set learning_enabled
        elif value is True:
            warnings.warn("Learning cannot be enabled for {} because it has no {}".
                  format(self.name, LearningMechanism.__name__))
            return

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    @beartype
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
            assert self.input_port.name != "Recurrent Input Port"
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
                                        activity_vector: Union[list, np.array],
                                        learning_function,
                                        learning_rate,
                                        learning_condition,
                                        matrix,
                                        ):

        learning_mechanism = AutoAssociativeLearningMechanism(default_variable=copy.deepcopy([activity_vector.defaults.value]),
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
                           learning_function: Optional[Callable] = None,
                           learning_rate: Optional[Union[numbers.Number, list, np.ndarray, np.matrix]] = None,
                           learning_condition: Union[Condition, TimeScale, Literal['UPDATE', 'CONVERGENCE']] = None,
                           context=None):
        """Provide user-accessible-interface to _instantiate_learning_mechanism

        Configure GRUMechanism for learning. Creates the following Components:

        * an `AutoAssociativeLearningMechanism` -- if the **learning_function** and/or **learning_rate** arguments are
          specified, they are used to construct the LearningMechanism, otherwise the values specified in the
          GRUMechanism's constructor are used;
        ..
        * a `MappingProjection` from the GRUMechanism's `primary OutputPort <OutputPort_Primary>`
          to the AutoAssociativeLearningMechanism's *ACTIVATION_INPUT* InputPort;
        ..
        * a `LearningProjection` from the AutoAssociativeLearningMechanism's *LEARNING_SIGNAL* OutputPort to
          the GRUMechanism's `recurrent_projection <GRUMechanism.recurrent_projection>`.

        """

        if not isinstance(learning_condition, Condition):
            if learning_condition == CONVERGENCE:
                learning_condition = WhenFinished(self)
            elif learning_condition == UPDATE:
                learning_condition = None

        try:
            shared_params = self.initial_shared_parameters['learning_mechanism']

            try:
                if learning_condition is None:
                    learning_condition = shared_params['learning_condition']
            except KeyError:
                pass

            try:
                if learning_rate is None:
                    learning_rate = shared_params['learning_rate']
            except KeyError:
                pass

            try:
                if learning_function is None:
                    learning_function = shared_params['learning_function']
            except KeyError:
                pass
        except KeyError:
            pass

        self.learning_mechanism = self._instantiate_learning_mechanism(activity_vector=self._learning_signal_source,
                                                                       learning_function=learning_function,
                                                                       learning_rate=learning_rate,
                                                                       learning_condition=learning_condition,
                                                                       matrix=self.recurrent_projection,
                                                                       )

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
        variable = self._parse_integrator_function_variable(variable, context=context)
        return super()._parse_function_variable(variable, context=context)

    def _parse_integrator_function_variable(self, variable, context=None):
        if self.has_recurrent_input_port:
            variable = self.combination_function.execute(variable=variable, context=context)

        return variable

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

    @property
    def _learning_signal_source(self):
        """Return default source of learning signal (`Primary OutputPort <OutputPort_Primary>)`
              Subclass can override this to provide another source (e.g., see `ContrastiveHebbianMechanism`)
        """
        return self.output_port

    def _get_state_ids(self):
        return super()._get_state_ids() + ["old_val"]

    def _get_state_struct_type(self, ctx):
        transfer_t = ctx.get_state_struct_type(super())
        return_t = ctx.get_output_struct_type(self)
        return pnlvm.ir.LiteralStructType([*transfer_t.elements, return_t])

    def _get_state_initializer(self, context):
        transfer_init = super()._get_state_initializer(context)

        # Initialize to OutputPort defaults.
        # That is what the recurrent projection finds.
        retval_init = (tuple(op.parameters.value.get(context)) if not np.isscalar(op.parameters.value.get(context)) else op.parameters.value.get(context) for op in self.output_ports)
        return (*transfer_init, tuple(retval_init))

    def _gen_llvm_function_reset(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" in tags

        # Check if we have reinitializers
        has_reinitializers_ptr = ctx.get_param_or_state_ptr(builder,
                                                            self,
                                                            "has_initializers",
                                                            param_struct_ptr=params)
        has_initializers = builder.load(has_reinitializers_ptr)
        not_initializers = builder.fcmp_ordered("==", has_initializers,
                                                has_initializers.type(0))
        with builder.if_then(not_initializers):
            builder.ret_void()

        # Reinit main function. This is a no-op if it's not a stateful function.
        reinit_func = ctx.import_llvm_function(self.function, tags=tags)
        reinit_params, reinit_state = ctx.get_param_or_state_ptr(builder,
                                                                 self,
                                                                 "function",
                                                                 param_struct_ptr=params,
                                                                 state_struct_ptr=state)
        reinit_in = builder.alloca(reinit_func.args[2].type.pointee, name="reinit_in")
        reinit_out = builder.alloca(reinit_func.args[3].type.pointee, name="reinit_out")
        builder.call(reinit_func, [reinit_params, reinit_state, reinit_in, reinit_out])

        # Reinit integrator function
        if self.integrator_mode:
            reinit_f = ctx.import_llvm_function(self.integrator_function, tags=tags)
            reinit_in = builder.alloca(reinit_f.args[2].type.pointee, name="integ_reinit_in")
            reinit_out = builder.alloca(reinit_f.args[3].type.pointee, name="integ_reinit_out")
            reinit_params, reinit_state = ctx.get_param_or_state_ptr(builder,
                                                                     self,
                                                                     "integrator_function",
                                                                     param_struct_ptr=params,
                                                                     state_struct_ptr=state)
            builder.call(reinit_f, [reinit_params, reinit_state, reinit_in, reinit_out])

        prev_val_ptr = ctx.get_param_or_state_ptr(builder, self, "old_val", state_struct_ptr=state)
        builder.store(prev_val_ptr.type.pointee(None), prev_val_ptr)
        return builder

    def _gen_llvm_input_ports(self, ctx, builder, params, state, arg_in):
        recurrent_params, recurrent_state = ctx.get_param_or_state_ptr(builder,
                                                                       self,
                                                                       "recurrent_projection",
                                                                       param_struct_ptr=params,
                                                                       state_struct_ptr=state)
        recurrent_f = ctx.import_llvm_function(self.recurrent_projection)

        # Extract the correct output port value
        prev_val_ptr = ctx.get_param_or_state_ptr(builder, self, "old_val", state_struct_ptr=state)
        recurrent_index = self.output_ports.index(self.recurrent_projection.sender)
        recurrent_in = builder.gep(prev_val_ptr, [ctx.int32_ty(0), ctx.int32_ty(recurrent_index)])

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

        assert not self.has_recurrent_input_port, "Configuration using combination function is not supported!"

        return super()._gen_llvm_input_ports(ctx, builder, params, state, arg_in)

    def _gen_llvm_output_ports(self, ctx, builder, value,
                               mech_params, mech_state, mech_in, mech_out):
        ret = super()._gen_llvm_output_ports(ctx, builder, value, mech_params,
                                             mech_state, mech_in, mech_out)

        prev_val_ptr = ctx.get_param_or_state_ptr(builder, self, "old_val", state_struct_ptr=mech_state)
        builder.store(builder.load(mech_out), prev_val_ptr)
        return ret
