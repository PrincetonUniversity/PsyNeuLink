# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  AutoAssociativeProjection ***********************************************

"""
.. _Auto_Associative_Overview:

Intro
-----

I am basically just a MappingProjection, except I'm intended to be used as a recurrent projection. I thus require
that my matrix (with which I multiply my input to produce my output) is a square matrix. By default I point to/from the
primary input state and output state of my owner. But you can specify the input and output state as well.
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
    Insert docs here
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
                 auto=None,
                 hetero=None,
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

        params = self._assign_args_to_param_dicts(auto=auto, hetero=hetero, function_params={MATRIX: matrix}, params=params)

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _instantiate_attributes_before_function(self, context=None):
        """
        """

        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Create self.matrix based on auto and hetero, if specified
        """

        super()._instantiate_attributes_after_function(context=context)

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

        # read auto and hetero from their ParameterStates, and put them into `auto_matrix` and `hetero_matrix`
        # (where auto_matrix is a diagonal matrix and hetero_matrix is a hollow matrix)
        raw_auto = owner_mech._parameter_states[AUTO].value
        auto_matrix = get_auto_matrix(raw_auto=raw_auto, size=owner_mech.size[0])
        if auto_matrix is None:
            raise AutoAssociativeError("The `auto` parameter of {} {} was invalid: it was equal to {}, and was of "
                                       "type {}. Instead, the `auto` parameter should be a number, 1D array, "
                                       "2d array, 2d list, or numpy matrix".
                                       format(owner_mech.__class__.__name__, owner_mech.name, raw_auto, type(raw_auto)))

        raw_hetero = owner_mech._parameter_states[HETERO].value
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

            # FIX: UPDATE FOR LEARNING BEGIN
            #    ??DELETE ONCE INTEGRATOR FUNCTION IS IMPLEMENTED
            # Pass params for ParameterState's function specified by instantiation in LearningProjection
            weight_change_params = matrix_parameter_state.paramsCurrent

            # Update parameter state: combines weightChangeMatrix from LearningProjection with matrix base_value
            matrix_parameter_state.update(weight_change_params, context=context)

            # Update MATRIX, and AUTO and HETERO accordingly
            self.matrix = matrix_parameter_state.value

            self.auto = np.diag(self.matrix).copy()
            self.hetero = self.matrix.copy()
            np.fill_diagonal(self.hetero, 0)

        return self.function(self.sender.value, params=params, context=context)


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
    #     if AUTO in owner_mech._parameter_states and hetero in owner_mech._parameter_states:
    #         owner_mech._parameter_states[AUTO].update(params=runtime_params, time_scale=time_scale, context=context + INITIALIZING)
    #         owner_mech._parameter_states[HETERO].update(params=runtime_params, time_scale=time_scale, context=context + INITIALIZING)
    #     else:
    #         raise AutoAssociativeError("Auto or Hetero ParameterState not found in {0} \"{1}\"; here are names of the "
    #                                    "current ParameterStates for {1}: {2}".format(owner_mech.__class__.__name__,
    #                                    owner_mech.name, owner_mech._parameter_states.key_values))

# a helper function that takes a specification of `hetero` and returns a hollow matrix with the right values
# (also used by RecurrentTransferMechanism.py)
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