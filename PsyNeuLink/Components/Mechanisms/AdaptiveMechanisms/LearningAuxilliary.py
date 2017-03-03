# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningAuxilliary ******************************************************

"""
.. _LearningAuxilliary_Overview:

Overview
--------

This module contains functions that automatically construct and compose the components necessary for learning
when this is specified for a process.

COMMENT:
    IMPLEMENT: LearningMechanism:
        PROCESS & SYSTEM:
          • Convert ProcessInputState and SystemInputState into Mechanisms with LinearFunction IDENTITY_FUNCTION
          • Use only one ObjectiveMechanism for all levels with the following args:
                default_input_value[[ACTIVITY][ERROR]]
                monitored_values: [[error_mech.OutputState][error_mech.objective_mechanism.OutputState]]
                names: [[ACTIVITY][ERROR]]
                function:  ErrorDerivative(variable, derivative)
                               variable[0] = activity
                               variable[1] = error_signal from error_mech ObjectiveMechanism (target for TERMINAL)
                               derivative = error_derivative (1 for TERMINAL)
                role:LEARNING
          • Use only one Learning mechanism with the following args:
                variable[[ACTIVATION_INPUT_INDEX][ACTIVATION_OUTPUT_INDEX][ERROR_SIGNAL_INDEX]
                activation_derivative
                error_matrix
                function
            Initialize and assign args with the following WIZZARD:
        WIZZARD:
            Needs to know
                activation_mech_output (Mechanism)
                    activation_derivative (function)
                error_mech (Mechanism)
                    error_derivative (function)
                    error_matrix (ndarray) - for MappingProjection from activation_mech_output to error_mech
            ObjectiveMechanism:
                Initialize variable:
                      use error_mech.outputState.valuee to initialize variable[ACTIVITY]
                      use outputState.value of error_mech's objective_mechanism to initialize variable[ERROR]
                Assign mapping projections:
                      nextLevel.outputState.value -> inputStates[ACTIVITY] of ObjectiveMechanism
                      nextLevel.objective_mechanism.outputState.value  -> inputStates[ERROR] of ObjectiveMechanism
                NOTE: For TERMINAL mechanism:
                          error_mech is Process or System InputState (function=Linear, so derivative =1), so that
                             error_mech.outputState.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Initialize variable:
                      use mapping_projection.sender.value to initialize variable[ACTIVATION_INPUT_INDEX]
                      use activation_mech_output_outputState.value to initialize variable[ACTIVATION_OUTPUT_INDEX]
                      use error_mech.objecdtive_mechanism.OutputState.value to initialize variable[ERROR_SIGNAL_INDEX]
                Assign activation_derivative using function of activation_mech_output of mapping_projection (one being learned)
                Assign error_derivative using function of error_mech
                Assign error_matrix as runtime_param using projection to error_mech [ALT: ADD TO VARIABLE]
                Assign mapping projections:
                      mapping_projection.sender -> inputStates[ACTIVATION_INPUT_INDEX] of LearningMechanism
                      activation_mech_output.outputState -> inputStates[ACTIVATION_OUTPUT_INDEX] of LearningMechanism
                      error_mech.objective_mechanism.OutputState.value -> inputStates[ERROR_SIGNAL_INDEX]

            For TARGET MECHANISM:  Matrix is IDENTITY MATRIX??
            For TARGET MECHANISM:  derivative for ObjectiveMechanism IDENTITY FUNCTION


    *************************************************************************

    Call in _instantiate_attributes_before_function() of LearningProjection

    Do the following:
        Get:
            activation_mech_projection (one being learned) (MappingProjection)
            activation_mech (ProcessingMechanism)
            activation_mech_input (OutputState)
            activation_mech_output (OutputState)
            activation_derivative (function)
            error_matrix (ParameterState)
            error_derivative (function)
            error_mech (error_source_mech) (ProcessingMechanism)
            error_signal_mech (LearningMechanism or ObjectiveMechanism)
            error_signal_mech_output (OutputState)
            error_objective_mech (error_source_objective_mech) (ObjectiveMechanism)

        Instantiate:
            ObjectiveMechanism:
                Construct with:
                   monitor_values: [error_mech.outputState.value, error_mech.objective_mech.outputState.value]
                   names = [ACTIVITY/SAMPLE, ERROR/TARGET]
                   function = ErrorDerivative(derivative=error_derivative)
                NOTE: For TERMINAL mechanism:
                          error_mech is Process or System InputState (function=Linear, so derivative =1), so that
                             error_mech.outputState.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Construct with:
                    variable=[[activation_mech_input],[activation_mech_output],[ObjectiveMechanism.outputState.value]]
                    error_matrix = error_matrix
                    function = one specified with Learning specification
                               NOTE: can no longer be specified as function for LearningProjection
                    names = [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]
                        NOTE:  this needs to be implemented for LearningMechanism as it is for ObjectiveMechanism
                Check that, if learning function expects a derivative (in user_params), that the one specified
                    is compatible with the function of the activation_mech
                Assign:
                    NOTE:  should do these in their own Learning module function, called by LearningMechanaism directly
                        as is done for ObjectiveMechanism
                    MappingProjection: activation_mech_projection.sender -> LearningMechanism.inputState[ACTIVATION_INPUT]
                    MappingProjection: activation_mech.outputState -> LearningMechanism.inputState[ACTIVATION_OUTPUT]
                    MappingProjection: ObjectiveMechanism -> LearningMechanism.inputState[ERROR_SIGNAL]


COMMENT


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

import numpy as np
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism, \
    _objective_mechanism_role
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism import LearningMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism import ACTIVATION_INPUT,\
    ACTIVATION_OUTPUT, ERROR_SIGNAL

from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec, _add_projection_from, _add_projection_to
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Functions.Function import Function, function_type, method_type
from PsyNeuLink.Components.Functions.Function import Linear, LinearCombination, Reinforcement, BackPropagation

TARGET_ERROR = "TARGET_ERROR"
TARGET_ERROR_MEAN = "TARGET_ERROR_MEAN"
TARGET_ERROR_SUM = "TARGET_ERROR_SUM"
TARGET_SSE = "TARGET_SSE"
TARGET_MSE = "TARGET_MSE"


class LearningAuxilliaryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return :keyword:`true` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
    Otherwise, return :keyword:`False`

    """
    if spec is LEARNING:
        return True
    else:
        return _is_projection_spec(spec)


def _instantiate_learning_components(learning_projection, context=None):
    """Instantiate learning components for a LearningProjection

    Instantiates a LearningMechanism or ObjectiveMechanism as the sender for each learning_projection in a learning
        sequence.  A learning sequence is defined as a sequence of ProcessingMechanisms, each of which has a
        projection — that has been specified for learning — to the next mechanism in the sequence.  This method
        instantiates the components required to support learning for those projections (most importantly,
        the LearningMechanisms that provide them with the learning_signal required to modify the matrix of the
        projection, and the ObjectiveMechanism that calculates the error_signal used to generate the learning_signals).


    It instantiates a LearningMechanism or ObjectiveMechanism as the sender for the learning_projection:
    - a LearningMechanism for projections to any Processing mechanism that is not the last in the learning sequence;
    - an ObjectiveMechanism for projections to a Processing mechanism that is the last in the learning sequence

    Assume that learning_projection's variable and parameters have been specified and validated,
       (which is the case when this method is called from the learning_projection itself in _instantiate_sender()).

    Notes:

    * Once the `receiver` for the learning_projection has been identified, or instantiated:
        - it is thereafter referred to (by reference to it owner) as the `activation_mech_projection`,
            and the mechanism to which it projects as the `activation_mech`;
        - the mechanism to which it projects is referred referred to as the error_mech (the source of the error_signal).

    * See learning_components class for the names of the components of learning used here.

    * This method supports only a single pathway for learning;  that is, the learning sequence must be a linear
        sequence of ProcessingMechanisms.  This is consistent with its implementation at the process level;
        convergent and divergent pathways for learning can be accomplished through compostion in a
        system.  Accordingly:

            - each LearningMechanism can have only one LearningProjection
            - each ProcessingMechanism can have only one MappingProjection that is subject to learning

      When searching downstream for projections that are being learned (to identify the error_mech mechanism as a
      source for the LearningMechanism's error_signal), the method uses the first projection that it finds, beginning
      in `primary outputState <OutputState_Primary>` of the activation_mech_output, and continues similarly
      in the error_mech.  If any conflicting implementations of learning components are encountered,
      an exception is raised.

    """
    # IMPLEMENTATION NOTE: CURRENT IMPLEMENTATION ONLY SUPPORTS CALL FROM LearningProjection._instantiate_sender();
    #                      IMPLEMENTATION OF SUPPORT FOR EXTERNAL CALLS:
    #                      - SHOULD CHECK UP FRONT WHETHER SENDER OR RECEIVER IS SPECIFIED, AND BRANCH ACCORDINGLY
    #                            FAILURE TO SPECIFY BOTH SHOULD RAISE AN EXCEPTION (OR FURTHER DEFER INSTANTIATION?)
    #                      - WILL REQUIRE MORE EXTENSIVE CHECKING AND VALIDATION
    #                              (E.G., OF WHETHER ANY LearningMechanisms IDENTIFIED HAVE A PROJECTION FROM AN
    #                               APPROPRIATE ObjectiveMechanism, etc.
    if not learning_projection.name in context:
        raise LearningAuxilliaryError("PROGRAM ERROR".format("_instantiate_learning_components only supports "
                                                             "calls from a LearningProjection._instantiate_sender()"))

    # First determine whether receiver for LearningProjection has been instantiated and, if not, instantiate it:
    #    this is required to instantiate the LearningMechanism, as well as an ObjectiveMechanism if necessary.
    try:
        isinstance(learning_projection.receiver.owner, MappingProjection)
    except AttributeError:
        _instantiate_receiver_for_learning_projection(learning_projection, context=context)
        # IMPLEMENTATION NOTE: THIS SHOULD IDENTIFY/CONFIRM RECEIVER BASED ON PROJECTIONS TO THE OBJECTIVE MECHANISM
        #                      OF THE learning_projection's LearningMechanism;  IF THERE ARE NONE, THEN RAISE EXCEPTION

    # Next, validate that the receiver does not receive any other LearningProjections
    # IMPLEMENTATION NOTE:  this may be overly restrictive in the context of a system --
    #                       will need to be dealt with in Composition implementation (by examining its learning graph??)
    if any((isinstance(projection, LearningProjection) and not projection is learning_projection) for projection in
           learning_projection.receiver.receivesFromProjections):
        raise LearningAuxilliaryError("{} can't be assigned as LearningProjection to {} since that already has one.".
                                      format(learning_projection.name, learning_projection.receiver.owner.name))

    # Now that the receiver has been identified, use it to get the components (lc) needed for learning.
    # Note: error-related components may not yet be defined (if LearningProjection is for a TERMINAL mechanism)
    lc = learning_components(learning_projection=learning_projection)

    # Check if activation_mech already has a projection to a LearningMechanism
    # IMPLEMENTATION NOTE: this uses the first projection found (starting with the primary outputState)

    for projection in lc.activation_mech_output.sendsToProjections:

        receiver_mech = projection.receiver.owner

        # Check if projection already projects to a LearningMechanism or ObjectiveMechanism
        if isinstance(receiver_mech, LearningMechanism):
            # If it projects to a LearningMechanism:
            #    it must project to the ACTIVATION_OUTPUT inputState of that LearningMechanism;
            #    the LearningMechanism must be the sender of learning_projection.
            if not (projection.receiver.name is ACTIVATION_OUTPUT and receiver_mech is learning_projection.sender.owner):
                if lc.activation_mech_projection.verbosePref:
                    warnings.warn("{} projects to a LearningMechanism ({}) that is not the sender of its "
                                  "LearningProjection ({})".
                                  format(lc.activation_mech.name,receiver_mech.name,learning_projection.name))
                continue

        if isinstance(receiver_mech, ObjectiveMechanism):
            # If it projects to an ObjectiveMechanism:
            #    it must project to the SAMPLE inputState of that ObjectiveMechanism;
            #    its role attribute must be set to LEARNING;
            #    it must also project to a LearningMechanism that is the sender for the learning_projection.
            if not (projection.receiver.name is SAMPLE and
                            LEARNING not in receiver_mech.role and
                            learning_projection.sender.owner is
                            receiver_mech.outputState.sendsToProjections[0].receiver.owner):
                if lc.activation_mech_projection.verbosePref:
                    warnings.warn("{} projects to an invalid ObjectiveMechanism ({})".
                                  format(lc.activation_mech.name, receiver_mech.name,learning_projection.name))
                    continue

        # activation_mech has a projection to a valid LearningMechanism or ObjectiveMechanism,
        #    so assign that as sender and return;
        # This assumes that:
        #    the LearningMechanism will be validated by learning_projection
        #        in its call to super()._instantiate_sender()
        #        (which is the case if this method is called from learning_projection._instantiate_sender()
        learning_projection.sender = receiver_mech.outputState
        return


    # Next, determine whether an ObjectiveMechanism or LearningMechanism should be assigned as the sender
    # It SHOULD be an ObjectiveMechanism (i.e., TARGET) if either:
    #     - it has either no outgoing projections or
    #     - it has not projections that receive a LearningProjection;
    #   in either case, lc.error_projection returns None.
    #   Note:  this assumes that LearningProjections are being assigned from the end of the pathway to the beginning.
    is_target = not lc.error_projection


    # INSTANTIATE learning function

    # Note: have to wait to do it here, as Backpropagation needs error_matrix,
    #       which depends on projection to ObjectiveMechanism

    learning_function = learning_projection.learning_function
    learning_rate = learning_projection.learning_rate

    # REINFORCEMENT LEARNING FUNCTION
    if learning_projection.componentName is RL_FUNCTION:

        # Force all activity and error arrays to be scalars
        activation_input = activation_output = error_output = error_signal  = np.array([0])

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     ACTIVATION FUNCTION AND/OR LEARNING RATE
        # learning_function = Reinforcement(variable=objective_mechanism.outputState.value,
        learning_function = Reinforcement(variable=[[activation_input],[activation_output],[error_signal]],
                                          activation_function=lc.activation_mech_fct,
                                          learning_rate=learning_rate)

    # BACKPROPAGATION LEARNING FUNCTION
    elif learning_function.componentName is BACKPROPAGATION_FUNCTION:

        # Get activation_mech values
        activation_input = np.zeros_like(lc.activation_mech_input.value)
        activation_output = np.zeros_like(lc.activation_mech_output.value)
        # Validate that the function for activation_mech has a derivative
        try:
            activation_derivative = lc.activation_mech_fct.derivative
        except AttributeError:
            raise LearningAuxilliaryError("Function for activation_mech of {} must have a derivative "
                                          "to be used with {}".
                                          format(self.name, BackPropagation.componentName))

        # Get error_mech values
        if is_target:
            error_output = np.ones_like(lc.activation_mech_output.value)
            error_signal = np.zeros_like(lc.activation_mech_output.value)
            error_matrix = np.identity(len(error_signal))
            # IMPLEMENTATION NOTE: Assign error_derivative to derivative of ProcessingInputState or SystemInputState
            #                      function when these are fully implemented as mechanisms
            error_derivative = Linear().derivative

        else:
            error_output = np.zeros_like(lc.error_mech_output.value)
            error_signal = np.zeros_like(lc.error_signal_mech_output.value)
            error_matrix = lc.error_matrix
            try:
                error_derivative = lc.error_mech_fct.derivative
            except AttributeError:
                raise LearningAuxilliaryError("Function for error_mech of {} must have a derivative "
                                              "to be used with {}".
                                              format(self.name, BackPropagation.componentName))

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     DERIVATIVE OR LEARNING_RATE
        learning_function = BackPropagation(variable_default=[activation_input,
                                                              activation_output,
                                                              error_output,
                                                              error_signal],
                                            activation_derivative_fct=activation_derivative,
                                            error_derivative_fct=error_derivative,
                                            error_matrix=error_matrix,
                                            learning_rate=learning_rate)

    else:
        raise LearningAuxilliaryError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                  format(learning_function.componentName, learning_projection.name))


    # INSTANTIATE ObjectiveMechanism

    # If it is a TARGET, instantiate an ObjectiveMechanism and assign as learning_projection's sender
    if is_target:

        # Instantiate ObjectiveMechanism
        # Notes:
        # * MappingProjections for ObjectiveMechanism's inputStates will be assigned in its own call to Composition
        # * Need to specify both default_input_value and monitored_values since they may not be the same
        #    sizes (e.g., for RL the monitored_value for the sample may be a vector, but its input_value must be scalar)
        # SAMPLE inputState for ObjectiveMechanism should come from activation_mech_output
        # TARGET inputState for ObjectiveMechanism should be specified by string (TARGET),
        #     so that it is left free to later be assigned a projection from ProcessInputState and/or SystemInputState
        # Assign derivative of Linear to lc.error_derivative (as default, until TARGET projection is assigned);
        #    this will induce a simple subtraction of target-sample (i.e., implement a comparator)

        sample_input = activation_output
        target_input = error_output
        # Assign outputStates for TARGET ObjectiveMechanism (used for reporting)
        object_mech_params = {OUTPUT_STATES:
                                  [{NAME:TARGET_ERROR},
                                   {NAME:TARGET_ERROR_MEAN,
                                    CALCULATE:lambda x: np.mean(x)},
                                   {NAME:TARGET_ERROR_SUM,
                                    CALCULATE:lambda x: np.sum(x)},
                                   {NAME:TARGET_SSE,
                                    CALCULATE:lambda x: np.sum(x*x)},
                                   {NAME:TARGET_MSE,
                                    CALCULATE:lambda x: np.sum(x*x)/len(x)}]}

        objective_mechanism = ObjectiveMechanism(default_input_value=[sample_input,
                                                                      target_input],
                                                 monitored_values=[lc.activation_mech_output,
                                                                   TARGET],
                                                 names=['SAMPLE','TARGET'],
                                                 function=LinearCombination(weights=[-1, 1]),
                                                 role=LEARNING,
                                                 params=object_mech_params,
                                                 name=lc.activation_mech_projection.name + " " + OBJECTIVE_MECHANISM)
        objective_mechanism.learning_role = TARGET

        try:
            lc.error_projection = objective_mechanism.inputState.receivesFromProjections[0]
            # FIX: THIS IS TO FORCE ASSIGNMENT (SINCE IT DOESN'T SEEM TO BE ASSIGNED BY TEST BELOW)
        except AttributeError:
            raise LearningAuxilliaryError("PROGRAM ERROR: problem finding projection to TARGET ObjectiveMechanism "
                                          "from {} when instantiating {}".
                                          format(lc.activation_mech.name, learning_projection.name))
        else:
            if not lc.error_matrix:
                raise LearningAuxilliaryError("PROGRAM ERROR: problem assigning error_matrix for projection to "
                                              "ObjectiveMechanism for {} when instantiating {}".
                                              format(lc.activation_mech.name, learning_projection.name))

# FIX: RETURN NONE FOR lc.error_mech IF IT IS AN OBJECTIVE MECHANISM
#      ?? DO TEST BELOWN ON ERROR_SIGNAL_MECH?
#      PUT CONDITIONAL BELOW ON WHETHER IT IS A TARGET (EXPECT ERROR SIGNAL MECH TO BE AN OBJECTIVE MECHANISM
#                                                        AND IF NOT A TARGET, THEN EXPECT A LEARNING MECHANISM


    # # Check that the required error-related learning_components have been assigned
    # if not (lc.error_mech and lc.error_mech_output and lc.error_derivative):
    #     raise LearningAuxilliaryError("PROGRAM ERROR:  not all error-related learning_components "
    #                                   "have been assigned for {}".format(learning_projection.name))

    # INSTANTIATE LearningMechanism

    # - LearningMechanism incoming projections (by inputState):
    #    ACTIVATION_INPUT:
    #        MappingProjection from activation_mech_input
    #    ACTIVATION_OUTPUT:
    #        MappingProjection from activation_mech_output
    #    ERROR_OUTPUT:
    #        is_target:
    #            [1...] size of error_output
    #        NOT is_target:
    #            MappingProjection from error_mech_output
    #    ERROR_SIGNAL:
    #        is_target:
    #            MappingProjection from error_signal_mech_output (ObjectiveMechanism.outputState)
    #        NOT is_target:
    #            MappingProjection from error_signal_mech_output (LearningMechanism.outputStates[ERROR_SIGNAL]

    learning_mechanism = LearningMechanism(variable=[activation_input,
                                                     activation_output,
                                                     error_output,
                                                     error_signal],
                                           function=learning_function,
                                           name = lc.activation_mech_projection.name + " " +LEARNING_MECHANISM)

    # Assign MappingProjection from activation_mech_input to LearningMechanism's ACTIVATION_INPUT inputState
    MappingProjection(sender=lc.activation_mech_input,
                      receiver=learning_mechanism.inputStates[ACTIVATION_INPUT],
                      matrix=IDENTITY_MATRIX)
    # Assign MappingProjection from activation_mech_output to LearningMechanism's ACTIVATION_OUTPUT inputState
    MappingProjection(sender=lc.activation_mech_output,
                      receiver=learning_mechanism.inputStates[ACTIVATION_OUTPUT],
                      matrix=IDENTITY_MATRIX)

    if is_target:
        # Assign array of 1's as the fixed value for LearningMechanism's ERROR_OUTPUT;
        #    this insures that error signal will remain a simple subtraction of TARGET-SAMPLE
        learning_mechanism.inputState[ERROR_OUTPUT].baseValue = np.ones_like(error_output)

    else:
        # Assign MappingProjection from error_mech_output to LearningMechanism's ERROR_OUTPUT inputState
        MappingProjection(sender=lc.error_mech_output,
                          receiver=learning_mechanism.inputStates[ERROR_OUTPUT],
                          matrix=IDENTITY_MATRIX)

    # Assign MappingProjection from source of error_signal to LearningMechanism's ERROR_SIGNAL inputState
    # Note:  the source of the error signal is set in lc.error_signal_mech:
    #     if is_target, this comes from the outputState of objective_mechanism;
    #     otherwise, it comes from outputStates[ERROR_SIGNAL] of the LearningMechanism for lc.error_mech
    MappingProjection(sender=error_signal_mech_output,
                      receiver=learning_mechanism.inputStates[ERROR_SIGNAL],
                      matrix=IDENTITY_MATRIX)

    # Assign learning_mechanism as sender of learning_projection and return
    # Note: learning_projection still has to be assigned to the learning_mechanism's outputState;
    #       however, this requires that it's variable be assigned (which occurs in the rest of its
    #       _instantiate_sender method, from which this was called) and that its value be assigned
    #       (which occurs in its _instantiate_function method).
    learning_projection.sender = learning_mechanism.outputState


class learning_components(object):
    """Gets components required to instantiate LearningMechanism and its Objective Function for a LearningProjection

    Has attributes for the following learning components relevant to a `LearningProjection`,
    each of which is found and/or validated if necessary before assignment:

    * `activation_mech_projection` (`MappingProjection`):  one being learned)
    * `activation_mech_input` (`OutputState`):  input to mechanism to which projection being learned projections
    * `activation_mech` (`ProcessingMechanism`):  mechanism to which projection being learned projects
    * `activation_mech_output` (`OutputState`):  output of activation_mech
    * `activation_mech_fct` (function):  function of mechanism to which projection being learned projects
    * `activation_derivative` (function):  derivative of function for activation_mech
    * `error_projection` (`MappingProjection`):  one that has the error_matrix
    * `error_matrix` (`ParameterState`):  parameterState for error_matrix
    * `error_derivative` (function):  deriviative of function of error_mech
    * `error_mech` (ProcessingMechanism):  mechanism to which error_projection projects
    * `error_mech_output` (OutputState):  outputState of error_mech, that projects either to the next
                                          ProcessingMechanism in the pathway, or to an ObjectiveMechanism
    * `error_signal_mech` (LearningMechanism or ObjectiveMechanism):  mechanism from which LearningMechanism
                                                                      gets its error_signal (ObjectiveMechanism for
                                                                      the last mechanism in a learning sequence; next
                                                                      LearningMechanism in the sequence for all others)
    * `error_signal_mech_output` (OutputState): outputState of error_signal_mech, that projects to the preceeding
                                                LearningMechanism in the learning sequence (or nothing for the 1st mech)
    """

    def __init__(self, learning_projection, context=None):

        self._validate_learning_projection(learning_projection)
        self.learning_projection = learning_projection

        self._activation_mech_projection = None
        self._activation_mech_input = None
        self._activation_mech = None
        self._activation_mech_output = None
        self._activation_mech_fct = None
        self._activation_derivative = None
        self._error_projection = None
        self._error_matrix = None
        self._error_derivative = None
        self._error_mech = None
        self._error_mech_output = None
        self._error_signal_mech = None
        self._error_signal_mech_output = None
        # self._error_objective_mech = None
        # self._error_objective_mech_output = None

        self.activation_mech_projection
        self.activation_mech_input
        self.activation_mech
        self.activation_mech_output
        self.activation_mech_fct
        self.activation_derivative
        self.error_projection
        self.error_matrix
        self.error_derivative
        self.error_mech
        self.error_mech_output
        self.error_signal_mech
        self.error_signal_mech_output
        # self.error_objective_mech
        # self.error_objective_mech_output


    def _validate_learning_projection(self, learning_projection):

        if not isinstance(learning_projection, LearningProjection):
            raise LearningAuxilliaryError("{} is not a LearningProjection".format(learning_projection.name))

    # HELPER METHODS FOR lc_components
    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_projection:  one being learned) (MappingProjection)
    @property
    def activation_mech_projection(self):
        def _get_act_proj():
            try:
                self.activation_mech_projection = self.learning_projection.receiver.owner
                return self.learning_projection.receiver.owner
            except AttributeError:
                raise LearningAuxilliaryError("activation_mech_projection not identified: learning_projection ({})"
                                              "not appear to have been assiged a receiver.".
                                              format(self.learning_projection))
        return self._activation_mech_projection or _get_act_proj()

    @activation_mech_projection.setter
    def activation_mech_projection(self, assignment):
        if isinstance(assignment, (MappingProjection)):
            self._activation_mech_projection = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech_projection; "
                                          "it must be a MappingProjection.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_input:  input to mechanism to which projection being learned projections (OutputState)
    @property
    def activation_mech_input(self):
        def _get_act_input():
            if not self.activation_mech_projection:
                return None
            try:
                self.activation_mech_input = self.activation_mech_projection.sender
                return self.activation_mech_projection.sender
            except AttributeError:
                raise LearningAuxilliaryError("activation_mech_input not identified: activation_mech_projection ({})"
                                              "not appear to have been assiged a sender.".
                                              format(self.activation_mech_projection))
        return self._activation_mech_input or _get_act_input()

    @activation_mech_input.setter
    def activation_mech_input(self, assignment):
        if isinstance(assignment, (OutputState)):
            self._activation_mech_input = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech_input; "
                                          "it must be a OutputState.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech:  mechanism to which projection being learned projects (ProcessingMechanism)
    @property
    def activation_mech(self):
        def _get_act_mech():
            if not self.activation_mech_projection:
                return None
            try:
                self.activation_mech = self.activation_mech_projection.receiver.owner
                return self.activation_mech_projection.receiver.owner
            except AttributeError:
                raise LearningAuxilliaryError("activation_mech not identified: activation_mech_projection ({})"
                                              "not appear to have been assiged a receiver.".
                                              format(self.learning_projection))
        return self._activation_mech or _get_act_mech()

    @activation_mech.setter
    def activation_mech(self, assignment):
        if isinstance(assignment, (ProcessingMechanism_Base)):
            self._activation_mech = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech; "
                                          "it must be a ProcessingMechanism.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_output:  output of activation_mech (OutputState)
    @property
    def activation_mech_output(self):
        def _get_act_sample():
            if not self.activation_mech:
                return None
            try:
                self.activation_mech_output = self.activation_mech.outputState
                return self.activation_mech.outputState
            except AttributeError:
                raise LearningAuxilliaryError("activation_mech_output not identified: activation_mech ({})"
                                              "not appear to have been assiged a primary outputState.".
                                              format(self.learning_projection))
        return self._activation_mech_output or _get_act_sample()

    @activation_mech_output.setter
    def activation_mech_output(self, assignment):
        if isinstance(assignment, (OutputState)):
            self._activation_mech_output =assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech_output; "
                                          "it must be a OutputState.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_fct:  function of mechanism to which projection being learned projects (function)
    @property
    def activation_mech_fct(self):
        def _get_act_mech_fct():
            if not self.activation_mech:
                return None
            try:
                self.activation_mech_fct = self.activation_mech.function_object
                return self.activation_mech.function_object
            except AttributeError:
                raise LearningAuxilliaryError("activation_mech_fct not identified: activation_mech ({})"
                                              "not appear to have been assiged a Function.".
                                              format(self.learning_projection))
        return self._activation_mech_fct or _get_act_mech_fct()

    @activation_mech_fct.setter
    def activation_mech_fct(self, assignment):
        if isinstance(assignment, (Function)):
            self._activation_mech_fct = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech_fct; "
                                          "it must be a Function.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_derivative:  derivative of function for activation_mech (function)
    @property
    def activation_derivative(self):
        def _get_act_deriv():
            if not self.activation_mech_fct:
                return None
            try:
                self._activation_derivative = self.activation_mech.function_object.derivative
                return self.activation_mech.function_object.derivative
            except AttributeError:
                raise LearningAuxilliaryError("activation_derivative not identified: activation_mech_fct ({})"
                                              "not appear to have a derivative defined.".
                                              format(self.learning_projection))
        return self._activation_derivative or _get_act_deriv()

    @activation_derivative.setter
    def activation_derivative(self, assignment):
        if isinstance(assignment, (function_type, method_type)):
            self._activation_derivative = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_derivative; "
                                          "it must be a function or method.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_projection:  one that has the error_matrix (MappingProjection)
    @property
    def error_projection(self):
        # Find from activation_mech:
        def _get_error_proj():
            if not self.activation_mech_output:
                return None
            projections = self.activation_mech_output.sendsToProjections
            # error_proj must be a MappingProjection that has a LearningProjection to it
            error_proj = next((projection for projection in projections if
                              (isinstance(projection, MappingProjection) and projection.has_learning_projection)),None)
            if not error_proj:
                return None
            self.error_projection = error_proj
            return error_proj
        return self._error_projection or _get_error_proj()

    @error_projection.setter
    def error_projection(self, assignment):
        if isinstance(assignment, (MappingProjection)):
            self._error_projection = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_projection; "
                                          "it must be a MappingProjection.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_matrix:  parameterState for error_matrix (ParameterState)
    # This must be found
    @property
    def error_matrix(self):

        # Find from error_projection
        def _get_err_matrix():
            if not self.error_projection:
                return None
            try:
                self.error_matrix = self.error_projection.parameterStates[MATRIX]
                return self.error_projection.parameterStates[MATRIX]
            except AttributeError:
                raise LearningAuxilliaryError("error_matrix not identified: error_projection ({})"
                                              "not not have a {} parameterState".
                                              format(self.error_projection))
        return self._error_matrix or _get_err_matrix()

    @error_matrix.setter
    def error_matrix(self, assignment):
        if isinstance(assignment, (ParameterState)):
            self._error_matrix = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_matrix; "
                                          "it must be a ParameterState.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_mech:  mechanism to which error_projection projects (ProcessingMechanism)
    @property
    def error_mech(self):

        # Find from error_projection
        def _get_err_mech():
            if not self.error_projection:
                return None
            try:
                self.error_mech = self.error_projection.receiver.owner
            except AttributeError:
                raise LearningAuxilliaryError("error_mech not identified: error_projection ({})"
                                              "does not appear to have a receiver or owner".
                                              format(self.error_projection))
            if not isinstance(self.error_mech, ProcessingMechanism_Base):
                raise LearningAuxilliaryError("error_mech found ({}) but it does not "
                                              "appear to be a ProcessingMechanism".
                                              format(self.error_mech.name))
            return self.error_projection.receiver.owner
        return self._error_mech or _get_err_mech()

    @error_mech.setter
    def error_mech(self, assignment):
        if isinstance(assignment, (ProcessingMechanism_Base)):
            self._error_mech = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_mech; "
                                          "it must be a ProcessingMechanism.")


    # -----------------------------------------------------------------------------------------------------------
    # error_derivative:  deriviative of function of error_mech (function)
    @property
    def error_derivative(self):
        # Find from error_mech:
        def _get_error_deriv():
            if not self.error_mech:
                return None
            try:
                self.error_derivative = self.error_mech.function_object.derivative
                return self.error_mech.function_object.derivative
            except AttributeError:
                raise LearningAuxilliaryError("error_derivative not identified: the function ({}) "
                                              "for error_mech ({}) does not have a derivative attribute".
                                              format(self.error_mech.function_object.__class__.__name__,
                                                     self.error_mech.name))
        return self._error_derivative or _get_error_deriv()

    @error_derivative.setter
    def error_derivative(self, assignment):
        if isinstance(assignment, (function_type, method_type)):
            self._error_derivative = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_derivative; "
                                          "it must be a function.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_mech_output: outputState of mechanism to which error_projection projects (OutputState)
    @property
    def error_mech_output(self):
        # Find from error_mech
        def _get_err_mech_out():
            if not self.error_mech:
                return None
            try:
                self.error_mech_output = self.error_mech.outputState
            except AttributeError:
                raise LearningAuxilliaryError("error_mech_output not identified: error_mech ({})"
                                              "does not appear to have an outputState".
                                              format(self.error_mech_output))
            if not isinstance(self.error_mech_output, OutputState):
                raise LearningAuxilliaryError("error_mech_output found ({}) but it does not "
                                              "appear to be an OutputState".
                                              format(self.error_mech_output.name))
            return self.error_mech.outputState
        return self._error_mech_output or _get_err_mech_out()

    @error_mech_output.setter
    def error_mech_output(self, assignment):
        if isinstance(assignment, (OutputState)):
            self._error_mech_output = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_mech_output; "
                                          "it must be an OutputState.")

    # ---------------------------------------------------------------------------------------------------------------
    # error_signal_mech:  learning mechanism for error_projection (LearningMechanism)
    @property
    def error_signal_mech(self):
        # Find from error_matrix:
        def _get_obj_mech():
            if not self.error_matrix:
                return None
            # search the projections to the error_matrix parameter state for a LearningProjection
            learning_proj = next((proj for proj in self.error_matrix.receivesFromProjections
                                 if isinstance(proj, LearningProjection)),None)
            # if there are none, the error_matrix might be for an error_projection to an ObjectiveMechanism
            #   (i.e., the TARGET mechanism)
            if not learning_proj:
                # if error_mech is the last in the learning sequence, then its error_matrix does not receive a
                #    LearningProjection, but its error_projection does project to an ObjectiveMechanism, so return that
                objective_mechanism = self.error_matrix.owner.receiver.owner
                if not isinstance(objective_mechanism, ObjectiveMechanism):
                    raise LearningAuxilliaryError("error_signal_mech not identified: error_matrix does not have "
                                                  "a LearningProjection and error_projection does not project to a "
                                                  "TARGET ObjectiveMechanism")
                else:
                    self.error_signal_mech = objective_mechanism
                    return self.error_signal_mech
            try:
                learning_mech = learning_proj.sender.owner
            except AttributeError:
                raise LearningAuxilliaryError("error_signal_mech not identified: "
                                              "the LearningProjection to error_matrix does not have a sender")
            if not isinstance(learning_mech, LearningMechanism):
                raise LearningAuxilliaryError("error_signal_mech not identified: "
                                              "the LearningProjection to error_matrix does not come from a "
                                              "LearningMechanism")
            self.error_signal_mech = learning_mech
            return self.error_signal_mech

        return self._error_signal_mech or _get_obj_mech()

    @error_signal_mech.setter
    def error_signal_mech(self, assignment):
        if assignment is None or isinstance(assignment, (LearningMechanism)):
            self._error_signal_mech = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_signal_mech; "
                                          "it must be a LearningMechanism.")

    # ---------------------------------------------------------------------------------------------------------------
    # FIX: MODIFY TO RETURN EITHER outputState if it is an ObjectiveMechanism or
    #                              outputStates[ERROR_SIGNAL] if it it a LearningMechanism
    # error_signal_mech_output: outputState of LearningMechanism for error_projection (OutputState)
    @property
    def error_signal_mech_output(self):
        # Find from error_mech
        def _get_err_sig_mech_out():
            if not self.error_signal_mech:
                return None
            if isinstance(self.error_signal_mech, ObjectiveMechanism):
                try:
                    self.error_signal_mech_output = self.error_signal_mech.outputState
                except AttributeError:
                    raise LearningAuxilliaryError("error_signal_mech_output not identified: error_signal_mech ({})"
                                                  "does not appear to have an outputState".
                                                  format(self.error_signal_mech.name))
                if not isinstance(self.error_signal_mech_output, OutputState):
                    raise LearningAuxilliaryError("error_signal_mech_output found ({}) for {} but it does not "
                                                  "appear to be an OutputState".
                                                  format(self.error_signal_mech.name,
                                                         self.error_signal_mech_output.name))
            elif isinstance(self.error_signal_mech, LearningMechanism):
                try:
                    self.error_signal_mech_output = self.error_signal_mech.outputStates[ERROR_SIGNAL]
                except AttributeError:
                    raise LearningAuxilliaryError("error_signal_mech_output not identified: error_signal_mech ({})"
                                                  "does not appear to have an ERROR_SIGNAL outputState".
                                                  format(self.error_signal_mech))
                if not isinstance(self.error_signal_mech_output, OutputState):
                    raise LearningAuxilliaryError("error_signal_mech_output found ({}) for {} but it does not "
                                                  "appear to be an OutputState".
                                                  format(self.error_signal_mech.name,
                                                         self.error_signal_mech_output.name))
            return self.error_signal_mech.outputState
        return self._error_signal_mech_output or _get_err_sig_mech_out()

    @error_signal_mech_output.setter
    def error_signal_mech_output(self, assignment):
        if isinstance(assignment, (OutputState)):
            self._error_signal_mech_output = assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_signal_mech_output; "
                                          "it must be an OutputState.")

    # # ---------------------------------------------------------------------------------------------------------------
    # # error_objective_mech:  TARGET objective mechanism for error_mech (ObjectiveMechanism)
    # @property
    # def error_objective_mech(self):
    #     # Find from error_matrix:
    #     def _get_obj_mech():
    #         if not self.error_matrix:
    #             return None
    #         learning_proj = next((proj for proj in self.error_matrix.receivesFromProjections
    #                              if isinstance(proj, LearningProjection)),None)
    #         if not learning_proj:
    #             # error_matrix is for a MappingProjection that projects to the TARGET ObjectiveMechanism, so return that
    #             if isinstance(self.error_matrix.owner.receiver.owner, ObjectiveMechanism):
    #                 self.error_objective_mech = self.error_matrix.owner.receiver.owner
    #                 return self.error_matrix.owner.receiver.owner
    #             else:
    #                 raise LearningAuxilliaryError("error_objective_mech not identified: error_matrix does not have a "
    #                                               "LearningProjection and is not for a TARGET ObjectiveMechanism")
    #         try:
    #             learning_mech = learning_proj.sender.owner
    #         except AttributeError:
    #             raise LearningAuxilliaryError("error_objective_mech not identified: "
    #                                           "the LearningProjection to error_matrix does not have a sender")
    #         if not isinstance(learning_mech, LearningMechanism):
    #             raise LearningAuxilliaryError("error_objective_mech not identified: "
    #                                           "the LearningProjection to error_matrix does not come from a "
    #                                           "LearningMechanism")
    #         try:
    #             error_obj_mech = next((proj.sender.owner
    #                                    for proj in learning_mech.inputStates[ERROR_SIGNAL].receivesFromProjections
    #                                    if isinstance(proj.sender.owner, ObjectiveMechanism)),None)
    #         except AttributeError:
    #             # return None
    #             raise LearningAuxilliaryError("error_objective_mech not identified: "
    #                                           "could not find any projections to the LearningMechanism ({})".
    #                                           format(learning_mech))
    #         # if not error_obj_mech:
    #         #     raise LearningAuxilliaryError("error_objective_mech not identified: "
    #         #                                   "the LearningMechanism ({}) does not receive a projection "
    #         #                                   "from an ObjectiveMechanism".
    #         #                                   format(learning_mech))
    #         self.error_objective_mech = error_obj_mech
    #         return error_obj_mech
    # 
    #     return self._error_objective_mech or _get_obj_mech()
    # 
    # @error_objective_mech.setter
    # def error_objective_mech(self, assignment):
    #     if assignment is None or isinstance(assignment, (ObjectiveMechanism)):
    #         self._error_objective_mech = assignment
    #     else:
    #         raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_objective_mech; "
    #                                       "it must be an ObjectiveMechanism.")
    # 
    # # ---------------------------------------------------------------------------------------------------------------
    # # error_objective_mech_output: outputState of ObjectiveMechanism for error_projection (ObjectiveMechanism)
    # @property
    # def error_objective_mech_output(self):
    #     # Find from error_mech
    #     def _get_err_obj_mech_out():
    #         if not self.error_objective_mech:
    #             return None
    #         try:
    #             self.error_objective_mech_output = self.error_objective_mech.outputState
    #         except AttributeError:
    #             raise LearningAuxilliaryError("error_objective_mech_output not identified: error_objective_mech ({})"
    #                                           "does not appear to have an outputState".
    #                                           format(self.error_objective_mech_output))
    #         if not isinstance(self.error_objective_mech_output, OutputState):
    #             raise LearningAuxilliaryError("error_objective_mech_output found ({}) but it does not "
    #                                           "appear to be an OutputState".
    #                                           format(self.error_objective_mech_output.name))
    #         return self.error_objective_mech.outputState
    #     return self._error_objective_mech_output or _get_err_obj_mech_out()
    # 
    # @error_objective_mech_output.setter
    # def error_objective_mech_output(self, assignment):
    #     if isinstance(assignment, (OutputState)):
    #         self._error_objective_mech_output = assignment
    #     else:
    #         raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to error_objective_mech_output; "
    #                                       "it must be an OutputState.")
