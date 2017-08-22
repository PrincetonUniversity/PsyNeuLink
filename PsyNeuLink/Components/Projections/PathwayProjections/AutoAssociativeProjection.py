# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  AutoAssociativeProjection ***********************************************

"""
.. _Auto_Associative_Overview:

Overview
--------

An AutoAssociativeProjection is a subclass of `MappingProjection` that acts as the recurrent projection for a
`RecurrentTransferMechanism`. The primary difference between an AutoAssociativeProjection and a basic MappingProjection
is that an AutoAssociativeProjection uses the `auto <RecurrentTransferMechanism.auto>` and
`hetero <RecurrentTransferMechanism.hetero>` parameters *on the RecurrentTransferMechanism* to help update its matrix:
this allows for a `ControlMechanism <ControlMechanism>` to control the `auto <RecurrentTransferMechanism.auto>` and
`hetero <RecurrentTransferMechanism.hetero>` parameters and thereby control the matrix.

.. _Auto_Associative_Creation:

Creating an AutoAssociativeProjection
-------------------------------------

An AutoAssociativeProjection is created automatically by a RecurrentTransferMechanism (or its subclasses), and is
stored as the `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` parameter of the
RecurrentTransferMechanism. It is not recommended to create an AutoAssociativeProjection on its own, because during
execution an AutoAssociativeProjection references parameters owned by its RecurrentTransferMechanism (see
`Execution <Auto_Associative_Execution>` below).

.. _Auto_Associative_Structure:

Auto Associative Structure
--------------------------

In structure, the AutoAssociativeProjection is almost identical to a MappingProjection: the only additional attributes
are `auto <AutoAssociativeProjection.auto>` and `hetero <AutoAssociativeProjection.hetero>`.

.. _Auto_Associative_Configurable_Attributes:

Configurable Attributes
~~~~~~~~~~~~~~~~~~~~~~~

Due to its specialized nature, most parameters of the AutoAssociativeProjection are not configurable: the `variable` is
determined by the format of the output of the RecurrentTransferMechanism, the `function` is always LinearMatrix, and so
on. The only configurable parameter is the matrix, configured through the **matrix**, **auto**, and/or **hetero**
arguments for a RecurrentTransferMechanism:

.. _Auto_Associative_Matrix:

* **matrix** - multiplied by the input to the AutoAssociativeProjection in order to produce the output. Specification of
  the **matrix**, **auto**, and/or **hetero** arguments determines the values of the matrix; **auto** determines the
  diagonal entries (representing the strength of the connection from each node to itself) and **hetero** determines
  the off-diagonal entries (representing connections between nodes).

.. _Auto_Associative_Execution:

Execution
---------

An AutoAssociativeProjection uses its `matrix <AutoAssociativeProjection.matrix>` parameter to transform the value of its
`sender <AutoAssociativeProjection.sender>`, and provide the result as input for its
`receiver <AutoAssociativeProjection.receiver>`, the primary input state of the RecurrentTransferMechanism.

.. note::
     During execution the AutoAssociativeProjection updates its `matrix <AutoAssociativeProjection.matrix> parameter
     based on the `auto <RecurrentTransferMechanism.auto>` and `hetero <RecurrentTransferMechanism.hetero>` parameters
     *on the `RecurrentTransferMechanism`*. (The `auto <AutoAssociativeProjection.auto>` and
     `hetero <AutoAssociativeProjection.hetero>` parameters of the AutoAssociativeProjection simply refer to their
     counterparts on the RecurrentTransferMechanism as well.) The reason for putting the `auto
     <RecurrentTransferMechanism.auto>` and `hetero <RecurrentTransferMechanism.hetero>` parameters on the
     RecurrentTransferMechanism is that this allows them to be modified by a `ControlMechanism <ControlMechanism>`.

.. _Auto_Associative_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Globals.Keywords import AUTO_ASSOCIATIVE_PROJECTION, DEFAULT_MATRIX, AUTO, HETERO, CHANGED
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Scheduling.TimeScale import CentralClock

parameter_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})
projection_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})

class AutoAssociativeError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

class AutoAssociativeProjection(MappingProjection):
    """
    AutoAssociativeProjection(                              \
        owner=None,                                         \
        sender=None,                                        \
        receiver=None,                                      \
        matrix=DEFAULT_MATRIX,                              \
        params=None,                                        \
        name=None,                                          \
        prefs=None                                          \
        context=None)

    Implements a MappingProjection that is self-recurrent on a `RecurrentTransferMechanism`; an AutoAssociativeProjection
    represents connections between nodes in a single-layer recurrent network. It multiplies the output of the
    `RecurrentTransferMechanism` by a matrix, then presents the product as input to the `RecurrentTransferMechanism`.

    Note: The reason **auto** and **hetero** are not arguments to the constructor of the AutoAssociativeProjection is
    because it is only ever created by a RecurrentTransferMechanism: by the time the AutoAssociativeProjection is
    created, the **auto** and **hetero** arguments are already incorporated into the **matrix** argument.

    Arguments
    ---------

    owner : Optional[Mechanism]
        simply specifies both the sender and receiver of the AutoAssociativeProjection. Setting owner=myMechanism is
        identical to setting sender=myMechanism and receiver=myMechanism.

    sender : Optional[OutputState or Mechanism]
        specifies the source of the Projection's input. If a Mechanism is specified, its
        `primary OutputState <OutputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    receiver : Optional[InputState or Mechanism]
        specifies the destination of the Projection's output.  If a Mechanism is specified, its
        `primary InputState <InputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        the matrix used by `function <AutoAssociativeProjection.function>` (default: `LinearCombination`) to transform
        the value of the `sender <AutoAssociativeProjection.sender>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default assignment (`LinearCombination`).  Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default AutoAssociativeProjection-<index>
        a string used for the name of the MappingProjection. When an AutoAssociativeProjection is created by a
        RecurrentTransferMechanism, its name is specified as "<name of RecurrentTransferMechanism> recurrent projection".

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the `PreferenceSet` for the MappingProjection.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see `PreferenceSet <LINK>` for details).

    Attributes
    ----------

    componentType : AUTO_ASSOCIATIVE_PROJECTION

    sender : OutputState
        identifies the source of the Projection's input.

    receiver: InputState
        identifies the destination of the Projection.

    learning_mechanism : LearningMechanism
        source of error signal for that determine changes to the `matrix <AutoAssociativeProjection.matrix>` when
        `learning <LearningProjection>` is used.

    matrix : 2d np.ndarray
        matrix used by `function <AutoAssociativeProjection.function>` to transform input from the
        `sender <MappingProjection.sender>` to the value provided to the `receiver <AutoAssociativeProjection.receiver>`.

    auto : number or 1d np.ndarray
        diagonal terms of the `matrix <AutoAssociativeProjection.matrix>` used by the AutoAssociativeProjection: if auto
        is a single number, it means the diagonal is uniform

    hetero : number or 2d np.ndarray
        off-diagonal terms of the `matrix <AutoAssociativeProjection.matrix>` used by the AutoAssociativeProjection: if
        hetero is a single number, it means the off-diagonal terms are all the same

    has_learning_projection : bool : False
        identifies whether the AutoAssociativeProjection's `MATRIX` `ParameterState <ParameterState>` has been assigned
        a `LearningProjection`.

    value : np.ndarray
        Output of AutoAssociativeProjection, transmitted to `variable <InputState.variable>` of `receiver`.

    name : str : default AutoAssociativeProjection-<index>
        a string used for the name of the MappingProjection. When an AutoAssociativeProjection is created by a
        RecurrentTransferMechanism, its name is specified as "<name of RecurrentTransferMechanism> recurrent projection".
        If not is specified, a default is assigned by `ProjectionRegistry`
        (see `Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for Projection.
        Specified in the **prefs** argument of the constructor for the Projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentType = AUTO_ASSOCIATIVE_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # necessary?
    paramClassDefaults = MappingProjection.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 sender=None,
                 receiver=None,
                 matrix=DEFAULT_MATRIX,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):

        if owner is not None:
            if not isinstance(owner, Mechanism):
                raise AutoAssociativeError('Owner of AutoAssociative Mechanism must either be None or a Mechanism')
            if sender is None:
                sender = owner
            if receiver is None:
                receiver = owner

        params = self._assign_args_to_param_dicts(function_params={MATRIX: matrix}, params=params)

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def execute(self, input=None, clock=CentralClock, time_scale=None, params=None, context=None):
        """
        Based heavily on the execute() method for MappingProjection.

        """

        # As of 7/21/17, modulation of parameters through ControlSignals is only possible on Mechanisms
        # so the ParameterStates for 'auto' and 'hetero' live on the RecurrentTransferMechanism rather than on
        # the AutoAssociativeProjection itself. So this projection must reference its owner's ParameterStates
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))

        param_keys = owner_mech._parameter_states.key_values
        if AUTO not in param_keys or HETERO not in param_keys:
            raise AutoAssociativeError("Auto or Hetero ParameterState not found in {0} \"{1}\"; here are names of the "
                                       "current ParameterStates for {1}: {2}".format(owner_mech.__class__.__name__,
                                                                                   owner_mech.name, param_keys))

        # update the param states for auto/hetero: otherwise, if they've changed since self's last execution, we won't
        # know because the mechanism may not have updated its param state yet (if we execute before the mechanism)
        self._update_auto_and_hetero(owner_mech, params, time_scale, context)

        # read auto and hetero from their ParameterStates, and put them into `auto_matrix` and `hetero_matrix`
        # (where auto_matrix is a diagonal matrix and hetero_matrix is a hollow matrix)
        raw_auto = owner_mech.auto
        auto_matrix = get_auto_matrix(raw_auto=raw_auto, size=owner_mech.size[0])
        if auto_matrix is None:
            raise AutoAssociativeError("The `auto` parameter of {} {} was invalid: it was equal to {}, and was of "
                                       "type {}. Instead, the `auto` parameter should be a number, 1D array, "
                                       "2d array, 2d list, or numpy matrix".
                                       format(owner_mech.__class__.__name__, owner_mech.name, raw_auto, type(raw_auto)))

        raw_hetero = owner_mech.hetero
        hetero_matrix = get_hetero_matrix(raw_hetero=raw_hetero, size=owner_mech.size[0])
        if hetero_matrix is None:
            raise AutoAssociativeError("The `hetero` parameter of {} {} was invalid: it was equal to {}, and was of "
                                       "type {}. Instead, the `hetero` parameter should be a number, 1D array of "
                                       "length one, 2d array, 2d list, or numpy matrix".
                                       format(owner_mech.__class__.__name__, owner_mech.name, raw_hetero, type(raw_hetero)))
        self.matrix = auto_matrix + hetero_matrix

        # note that updating parameter states MUST happen AFTER self.matrix is set by auto_matrix and hetero_matrix,
        # because setting self.matrix only changes the previous_value/variable of the 'matrix' parameter state (which
        # holds the matrix parameter) and the matrix parameter state must be UPDATED AFTERWARDS to put the new value
        # from the previous_value into the value of the parameterState
        self._update_parameter_states(runtime_params=params, time_scale=time_scale, context=context)

        # Check whether error_signal has changed
        if self.learning_mechanism and self.learning_mechanism.status == CHANGED:

            # Assume that if learning_mechanism attribute is assigned,
            #    both a LearningProjection and ParameterState[MATRIX] to receive it have been instantiated
            matrix_parameter_state = self._parameter_states[MATRIX]

            # Assign current MATRIX to parameter state's base_value, so that it is updated in call to execute()
            setattr(self, '_'+MATRIX, self.matrix)

            # Update MATRIX, and AUTO and HETERO accordingly
            self.matrix = matrix_parameter_state.value

            owner_mech.auto = np.diag(self.matrix).copy()
            owner_mech.hetero = self.matrix.copy()
            np.fill_diagonal(owner_mech.hetero, 0)

        return self.function(self.sender.value, params=params, context=context)

    def _update_auto_and_hetero(self, owner_mech=None, runtime_params=None, time_scale=TimeScale.TRIAL, context=None):
        if owner_mech is None:
            if isinstance(self.sender, OutputState):
                owner_mech = self.sender.owner
            elif isinstance(self.sender, Mechanism):
                owner_mech = self.sender
            else:
                raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                           " the sender is {}".
                                           format(self.__class__.__name__, self.name, self.sender))
        if AUTO in owner_mech._parameter_states and HETERO in owner_mech._parameter_states:
            owner_mech._parameter_states[AUTO].update(params=runtime_params, time_scale=time_scale,
                                                      context=context + INITIALIZING)
            owner_mech._parameter_states[HETERO].update(params=runtime_params, time_scale=time_scale,
                                                        context=context + INITIALIZING)


    # NOTE 7/25/17 CW: Originally, this override was written because if the user set the 'auto' parameter on the
        # recurrent mechanism, the parameter state wouldn't update until after the mechanism executed: since the system
        # first runs the projection, then runs the mechanism, the projection initially uses the 'old' value. However,
        # this is commented out because this may in fact be the desired behavior.
        # Two possible solutions: allow control to be done on projections, or build a more general way to allow
        # projections to read parameters from mechanisms.
    # def _update_parameter_states(self, runtime_params=None, time_scale=None, context=None):
    #     """Update this projection's owner mechanism's `auto` and `hetero` parameter states as well! The owner mechanism
    #     should be a RecurrentTransferMechanism, which DOES NOT update its own `auto` and `hetero` parameter states during
    #     its _update_parameter_states function (so that the ParameterState is not redundantly updated).
    #     Thus, if you want to have an AutoAssociativeProjection on a mechanism that's not a RecurrentTransferMechanism,
    #     your mechanism must similarly exclude `auto` and `hetero` from updating.
    #     """
    #     super()._update_parameter_states(runtime_params, time_scale, context)
    #
    #     if isinstance(self.sender, OutputState):
    #         owner_mech = self.sender.owner
    #     elif isinstance(self.sender, Mechanism):
    #         owner_mech = self.sender
    #     else:
    #         raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently the"
    #                                    " sender is {}".
    #                                    format(self.__class__.__name__, self.name, self.sender))
    #
    #     if AUTO in owner_mech._parameter_states and HETERO in owner_mech._parameter_states:
    #         owner_mech._parameter_states[AUTO].update(params=runtime_params, time_scale=time_scale, context=context + INITIALIZING)
    #         owner_mech._parameter_states[HETERO].update(params=runtime_params, time_scale=time_scale, context=context + INITIALIZING)
    #     else:
    #         raise AutoAssociativeError("Auto or Hetero ParameterState not found in {0} \"{1}\"; here are names of the "
    #                                    "current ParameterStates for {1}: {2}".format(owner_mech.__class__.__name__,
    #                                    owner_mech.name, owner_mech._parameter_states.key_values))

    # these properties allow the auto and hetero properties to live purely on the RecurrentTransferMechanism
    @property
    def auto(self):
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))
        return owner_mech.auto

    @auto.setter
    def auto(self, setting):
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))
        owner_mech.auto = setting

    @property
    def hetero(self):
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))
        return owner_mech.hetero

    @hetero.setter
    def hetero(self, setting):
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))
        owner_mech.hetero = setting

    @property
    def matrix(self):
        return super(AutoAssociativeProjection, self.__class__).matrix.fget(self)

    @matrix.setter
    def matrix(self, setting):
        super(AutoAssociativeProjection, self.__class__).matrix.fset(self, setting)
        if isinstance(self.sender, OutputState):
            owner_mech = self.sender.owner
        elif isinstance(self.sender, Mechanism):
            owner_mech = self.sender
        else:
            raise AutoAssociativeError("The sender of the {} \'{}\' must be a Mechanism or OutputState: currently"
                                       " the sender is {}".
                                       format(self.__class__.__name__, self.name, self.sender))
        mat_setting = np.array(setting).copy()
        owner_mech.auto = np.diag(setting).copy()
        np.fill_diagonal(mat_setting, 0)
        owner_mech.hetero = mat_setting

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
        return np.diag(np.full(size, raw_auto))
    elif ((isinstance(raw_auto, np.ndarray) and raw_auto.ndim == 1) or
              (isinstance(raw_auto, list) and np.array(raw_auto).ndim == 1)):
        if len(raw_auto) == 1:
            return np.diag(np.full(size, raw_auto[0]))
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