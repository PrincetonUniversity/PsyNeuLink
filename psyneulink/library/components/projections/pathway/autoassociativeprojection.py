# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  AutoAssociativeProjection ***********************************************

"""

Contents
--------

  * `AutoAssociative_Overview`
  * `AutoAssociative_Creation`
  * `AutoAssociative_Structure`
      - `AutoAssociative_Configurable_Attributes`
  * `AutoAssociative_Execution`
  * `AutoAssociative_Class_Reference`


.. _AutoAssociative_Overview:

Overview
--------

An AutoAssociativeProjection is a subclass of `MappingProjection` that acts as the recurrent projection for a
`RecurrentTransferMechanism`. The primary difference between an AutoAssociativeProjection and a basic MappingProjection
is that an AutoAssociativeProjection uses the `auto <RecurrentTransferMechanism.auto>` and
`hetero <RecurrentTransferMechanism.hetero>` parameters *on the RecurrentTransferMechanism* to help update its matrix:
this allows for a `ControlMechanism <ControlMechanism>` to control the `auto <RecurrentTransferMechanism.auto>` and
`hetero <RecurrentTransferMechanism.hetero>` parameters and thereby control the matrix.

AutoAssociativeProjection represents connections between nodes in a single-layer recurrent network. It multiplies
the output of the `RecurrentTransferMechanism` by a matrix, then presents the product as input to the
`RecurrentTransferMechanism`.



.. _AutoAssociative_Creation:

Creating an AutoAssociativeProjection
-------------------------------------

An AutoAssociativeProjection is created automatically by a RecurrentTransferMechanism (or its subclasses), and is
stored as the `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` parameter of the
RecurrentTransferMechanism. It is not recommended to create an AutoAssociativeProjection on its own, because during
execution an AutoAssociativeProjection references parameters owned by its RecurrentTransferMechanism (see
`Execution <AutoAssociative_Execution>` below).

.. _AutoAssociative_Structure:

Auto Associative Structure
--------------------------

In structure, the AutoAssociativeProjection is almost identical to a MappingProjection: the only additional attributes
are `auto <AutoAssociativeProjection.auto>` and `hetero <AutoAssociativeProjection.hetero>`.

.. _AutoAssociative_Configurable_Attributes:

*Configurable Attributes*
~~~~~~~~~~~~~~~~~~~~~~~~~

Due to its specialized nature, most parameters of the AutoAssociativeProjection are not configurable: the `variable` is
determined by the format of the output of the RecurrentTransferMechanism, the `function` is always LinearMatrix, and so
on. The only configurable parameter is the matrix, configured through the **matrix**, **auto**, and/or **hetero**
arguments for a RecurrentTransferMechanism:

.. _AutoAssociative_Matrix:

* **matrix** - multiplied by the input to the AutoAssociativeProjection in order to produce the output. Specification of
  the **matrix**, **auto**, and/or **hetero** arguments determines the values of the matrix; **auto** determines the
  diagonal entries (representing the strength of the connection from each node to itself) and **hetero** determines
  the off-diagonal entries (representing connections between nodes).

.. _AutoAssociative_Execution:

Execution
---------

An AutoAssociativeProjection uses its `matrix <AutoAssociativeProjection.matrix>` parameter to transform the value of
its `sender <AutoAssociativeProjection.sender>`, and provide the result as input for its
`receiver <AutoAssociativeProjection.receiver>`, the primary InputPort of the RecurrentTransferMechanism.

.. note::
     During execution the AutoAssociativeProjection updates its `matrix <AutoAssociativeProjection.matrix> parameter
     based on the `auto <RecurrentTransferMechanism.auto>` and `hetero <RecurrentTransferMechanism.hetero>` parameters
     *on the `RecurrentTransferMechanism`*. (The `auto <AutoAssociativeProjection.auto>` and
     `hetero <AutoAssociativeProjection.hetero>` parameters of the AutoAssociativeProjection simply refer to their
     counterparts on the RecurrentTransferMechanism as well.) The reason for putting the `auto
     <RecurrentTransferMechanism.auto>` and `hetero <RecurrentTransferMechanism.hetero>` parameters on the
     RecurrentTransferMechanism is that this allows them to be modified by a `ControlMechanism <ControlMechanism>`.

.. _AutoAssociative_Class_Reference:

Class Reference
---------------

"""
import numbers

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.transferfunctions import LinearMatrix
from psyneulink.core.components.functions.function import get_matrix
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import projection_keywords
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import AUTO_ASSOCIATIVE_PROJECTION, DEFAULT_MATRIX, HOLLOW_MATRIX, MATRIX
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'AutoAssociativeError', 'AutoAssociativeProjection', 'get_auto_matrix', 'get_hetero_matrix',
]

parameter_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})
projection_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})


class AutoAssociativeError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _matrix_getter(owning_component=None, context=None):
    return owning_component.owner_mech.parameters.matrix._get(context)


def _matrix_setter(value, owning_component=None, context=None):
    owning_component.owner_mech.parameters.matrix._set(value, context)
    return value


def _auto_getter(owning_component=None, context=None):
    return owning_component.owner_mech.parameters.auto._get(context)


def _auto_setter(value, owning_component=None, context=None):
    owning_component.owner_mech.parameters.auto._set(value, context)
    return value


def _hetero_getter(owning_component=None, context=None):
    return owning_component.owner_mech.parameters.hetero._get(context)


def _hetero_setter(value, owning_component=None, context=None):
    owning_component.owner_mech.parameters.hetero._set(value, context)
    return value


class AutoAssociativeProjection(MappingProjection):
    """
    AutoAssociativeProjection(
        )

    Subclass of `MappingProjection` that is self-recurrent on a `RecurrentTransferMechanism`.
    See `MappingProjection <MappingProjection_Class_Reference>` and `Projection <Projection_Class_Reference>`
    for additional arguments and attributes.

    COMMENT:
        JDC [IN GENERAL WE HAVE TRIED TO DISCOURAGE SUCH DEPENDENCIES;  BETTER TO HAVE IT ACCEPT ARGUMENTS THAT
        RecurrentTransferMechanism (or Composition) PROVIDES THAN ASSUME THEY ARE ON ANOTHER OBJECT THAT CREATED
        THIS ONE]
    Note: The reason **auto** and **hetero** are not arguments to the constructor of the AutoAssociativeProjection is
    because it is only ever created by a RecurrentTransferMechanism: by the time the AutoAssociativeProjection is
    created, the **auto** and **hetero** arguments are already incorporated into the **matrix** argument.

    COMMENT

    Arguments
    ---------

    sender : OutputPort or Mechanism : default None
        specifies the source of the Projection's input; must be (or belong to) the same Mechanism as **receiver**,
        and the length of its `value <OutputPort.value>` must match that of the `variable <InputPort.variable>` of
        the **receiver**.

    receiver: InputPort or Mechanism : default None
        specifies the destination of the Projection's output; must be (or belong to) the same Mechanism as **sender**,
        and the length of its `variable <InputPort.variable>` must match the `value <OutputPort.value>` of **sender**.

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        specifies the matrix used by `function <Projection_Base.function>` (default: `LinearCombination`) to
        transform the `value <Projection_Base.value>` of the `sender <MappingProjection.sender>` into a value
        provided to the `variable <InputPort.variable>` of the `receiver <MappingProjection.receiver>` `InputPort`;
        must be a square matrix (i.e., have the same number of rows and columns).

    Attributes
    ----------

    sender : OutputPort
        the `OutputPort` of the `Mechanism <Mechanism>` that is the source of the Projection's input; in the case of
        an AutoAssociativeProjection, it is an OutputPort of the same Mechanism to which the `receiver
        <AutoAssociativeProjection.receiver>` belongs.

    receiver: InputPort
        the `InputPort` of the `Mechanism <Mechanism>` that is the destination of the Projection's output; in the case
        of an AutoAssociativeProjection, it is an InputPort of the same Mechanism to which the `sender
        <AutoAssociativeProjection.sender>` belongs.

    matrix : 2d np.ndarray
        square matrix used by `function <AutoAssociativeProjection.function>` to transform input from the `sender
        <MappingProjection.sender>` to the value provided to the `receiver <AutoAssociativeProjection.receiver>`;
        in the case of an AutoAssociativeProjection.

    """

    componentType = AUTO_ASSOCIATIVE_PROJECTION
    className = componentType
    suffix = " " + className

    class Parameters(MappingProjection.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <AutoAssociativeProjection.variable>`

                    :default value: numpy.array([[0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                auto
                    see `auto <AutoAssociativeProjection.auto>`

                    :default value: 1
                    :type: ``int``

                function
                    see `function <AutoAssociativeProjection.function>`

                    :default value: `LinearMatrix`
                    :type: `Function`

                hetero
                    see `hetero <AutoAssociativeProjection.hetero>`

                    :default value: 0
                    :type: ``int``

                matrix
                    see `matrix <AutoAssociativeProjection.matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: ``str``
        """
        variable = Parameter(np.array([[0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        # function is always LinearMatrix that requires 1D input
        function = Parameter(LinearMatrix, stateful=False, loggable=False)

        auto = Parameter(1, getter=_auto_getter, setter=_auto_setter, modulable=True)
        hetero = Parameter(0, getter=_hetero_getter, setter=_hetero_setter, modulable=True)
        matrix = Parameter(DEFAULT_MATRIX, getter=_matrix_getter, setter=_matrix_setter, modulable=True)

    classPreferenceLevel = PreferenceLevel.TYPE

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 sender=None,
                 receiver=None,
                 matrix=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs
                 ):

        if owner is not None:
            if not isinstance(owner, Mechanism):
                raise AutoAssociativeError('Owner of AutoAssociative Mechanism must either be None or a Mechanism')
            if sender is None:
                sender = owner
            if receiver is None:
                receiver = owner

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    # COMMENTED OUT BY KAM 1/9/2018 -- this method is not currently used; should be moved to Recurrent Transfer Mech
    #     if it is used in the future

    # def _update_auto_and_hetero(self, owner_mech=None, runtime_params=None, time_scale=TimeScale.TRIAL, context=None):
    #     if owner_mech is None:
    #         if isinstance(self.sender, OutputPort):
    #             owner_mech = self.sender.owner
    #         elif isinstance(self.sender, Mechanism):
    #             owner_mech = self.sender
    #         else:
    #             raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputPort: currently"
    #                                        " the sender is {}".
    #                                        format(self.__class__.__name__, self.name, self.sender))
    #     if AUTO in owner_mech._parameter_ports and HETERO in owner_mech._parameter_ports:
    #         owner_mech._parameter_ports[AUTO].update(context=context, params=runtime_params, time_scale=time_scale)
    #         owner_mech._parameter_ports[HETERO].update(context=context, params=runtime_params, time_scale=time_scale)
    #

    # END OF COMMENTED OUT BY KAM 1/9/2018

    # NOTE 7/25/17 CW: Originally, this override was written because if the user set the 'auto' parameter on the
        # recurrent mechanism, the ParameterPort wouldn't update until after the mechanism executed: since the system
        # first runs the projection, then runs the mechanism, the projection initially uses the 'old' value. However,
        # this is commented out because this may in fact be the desired behavior.
        # Two possible solutions: allow control to be done on projections, or build a more general way to allow
        # projections to read parameters from mechanisms.
    # def _update_parameter_ports(self, runtime_params=None, context=None):
    #     """Update this projection's owner mechanism's `auto` and `hetero` parameter ports as well! The owner mechanism
    #     should be a RecurrentTransferMechanism, which DOES NOT update its own `auto` and `hetero` parameter ports during
    #     its _update_parameter_ports function (so that the ParameterPort is not redundantly updated).
    #     Thus, if you want to have an AutoAssociativeProjection on a mechanism that's not a RecurrentTransferMechanism,
    #     your mechanism must similarly exclude `auto` and `hetero` from updating.
    #     """
    #     super()._update_parameter_ports(runtime_params, context)
    #
    #     if isinstance(self.sender, OutputPort):
    #         owner_mech = self.sender.owner
    #     elif isinstance(self.sender, Mechanism):
    #         owner_mech = self.sender
    #     else:
    #         raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputPort: currently the"
    #                                    " sender is {}".
    #                                    format(self.__class__.__name__, self.name, self.sender))
    #
    #     if AUTO in owner_mech._parameter_ports and HETERO in owner_mech._parameter_ports:
    #         owner_mech._parameter_ports[AUTO].update(context=context, params=runtime_params)
    #         owner_mech._parameter_ports[HETERO].update(context=context, params=runtime_params)
    #     else:
    #         raise AutoAssociativeError("Auto or Hetero ParameterPort not found in {0} \"{1}\"; here are names of the "
    #                                    "current ParameterPorts for {1}: {2}".format(owner_mech.__class__.__name__,
    #                                    owner_mech.name, owner_mech._parameter_ports.key_values))

    @property
    def owner_mech(self):
        if isinstance(self.sender, OutputPort):
            return self.sender.owner
        elif isinstance(self.sender, Mechanism):
            return self.sender
        else:
            raise AutoAssociativeError(
                "The sender of the {} \'{}\' must be a Mechanism or OutputPort: currently the sender is {}".format(
                    self.__class__.__name__, self.name, self.sender
                )
            )

    # these properties allow the auto and hetero properties to live purely on the RecurrentTransferMechanism
    @property
    def auto(self):
        return self.owner_mech.auto

    @auto.setter
    def auto(self, setting):
        self.owner_mech.auto = setting

    @property
    def hetero(self):
        return self.owner_mech.hetero

    @hetero.setter
    def hetero(self, setting):
        self.owner_mech.hetero = setting

    @property
    def matrix(self):
        owner_mech = self.owner_mech
        if hasattr(owner_mech, "matrix"):
            return owner_mech.matrix
        return super(AutoAssociativeProjection, self.__class__).matrix.fget(self)

    @matrix.setter
    def matrix(self, setting):
        owner_mech = self.owner_mech
        if hasattr(owner_mech, "matrix"):
            owner_mech.matrix = setting
        else:
            super(AutoAssociativeProjection, self.__class__).matrix.fset(self, setting)


# a helper function that takes a specification of `hetero` and returns a hollow matrix with the right values
def get_hetero_matrix(raw_hetero, size):
    if isinstance(raw_hetero, numbers.Number):
        return get_matrix(HOLLOW_MATRIX, size, size) * raw_hetero
    elif ((isinstance(raw_hetero, np.ndarray) and raw_hetero.ndim == 1) or
              (isinstance(raw_hetero, list) and np.array(raw_hetero).ndim == 1)):
        if len(raw_hetero) != 1:
            return None
        return get_matrix(HOLLOW_MATRIX, size, size) * raw_hetero[0]
    elif (isinstance(raw_hetero, np.matrix) or
              (isinstance(raw_hetero, np.ndarray) and raw_hetero.ndim == 2) or
              (isinstance(raw_hetero, list) and np.array(raw_hetero).ndim == 2)):
        np.fill_diagonal(raw_hetero, 0)
        return np.array(raw_hetero)
    else:
        return None


# similar to get_hetero_matrix() above
def get_auto_matrix(raw_auto, size):
    if isinstance(raw_auto, numbers.Number):
        return np.diag(np.full(size, raw_auto, dtype=np.float))
    elif ((isinstance(raw_auto, np.ndarray) and raw_auto.ndim == 1) or
              (isinstance(raw_auto, list) and np.array(raw_auto).ndim == 1)):
        if len(raw_auto) == 1:
            return np.diag(np.full(size, raw_auto[0], dtype=np.float))
        else:
            if len(raw_auto) != size:
                return None
            return np.diag(raw_auto)
    elif (isinstance(raw_auto, np.matrix) or
              (isinstance(raw_auto, np.ndarray) and raw_auto.ndim == 2) or
              (isinstance(raw_auto, list) and np.array(raw_auto).ndim == 2)):
        # we COULD add a validation here to ensure raw_auto is diagonal, but it would slow stuff down.
        return np.array(raw_auto)
    else:
        return None
