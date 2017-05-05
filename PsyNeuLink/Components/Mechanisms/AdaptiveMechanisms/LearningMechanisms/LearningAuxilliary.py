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
                      nextLevel.outputState.value -> input_states[ACTIVITY] of ObjectiveMechanism
                      nextLevel.objective_mechanism.outputState.value  -> input_states[ERROR] of ObjectiveMechanism
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
                      mapping_projection.sender -> input_states[ACTIVATION_INPUT_INDEX] of LearningMechanism
                      activation_mech_output.outputState -> input_states[ACTIVATION_OUTPUT_INDEX] of LearningMechanism
                      error_mech.objective_mechanism.OutputState.value -> input_states[ERROR_SIGNAL_INDEX]

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
                    MappingProjection: activation_mech_projection.sender -> LearningMechanism.input_state[ACTIVATION_INPUT]
                    MappingProjection: activation_mech.outputState -> LearningMechanism.input_state[ACTIVATION_OUTPUT]
                    MappingProjection: ObjectiveMechanism -> LearningMechanism.input_state[ERROR_SIGNAL]


COMMENT


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

import numpy as np

from PsyNeuLink.Components.Functions.Function import Function, function_type, method_type
from PsyNeuLink.Components.Functions.Function import Linear, LinearCombination, Reinforcement, BackPropagation
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import ACTIVATION_INPUT, \
    ACTIVATION_OUTPUT, ERROR_SIGNAL
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Globals.Keywords import *

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

    * See LearningComponents class for the names of the components of learning used here.

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
    lc = LearningComponents(learning_projection=learning_projection)

    # Check if activation_mech already has a projection to an ObjectiveMechanism or a LearningMechanism
    #    (i.e., the source of an error_signal for learning_projection)
    # IMPLEMENTATION NOTE: this uses the first projection found (starting with the primary outputState)

    # FIX: CHECK THAT THE PROJECTION IS TO ANOTHER MECHANISM IN THE CURRENT PROCESS;
    #      OTHERWISE, ALLOW OBJECTIVE_MECHANISM TO BE IMPLEMENTED


    # Check whether lc.activation_mech belongs to more than one process and, if it does, it is the TERMINAL
    #    the one currently being instantiated, in which case it needs to be assigned an ObjectiveMechanism
    #
    # If the only MappingProjections from lc.activation_mech are to mechanisms that are not in the same
    #    process as the mechanism that projects to it (i.e., lc.activation_mech_input.owner), then:
    #    - set is_target to True so that it will be assigned an ObjectiveMechanism
    # Note: this handles the case in which the current mech belongs to more than one process, and is the
    #    the TERMINAL of the one currently being instantiated.
    # IMPLEMENTATION NOTE:  could check whether current mech is a TERMINAL for the process currently being
    #                       instantiated, identified by the mechanism that projects to it.  However, the latter
    #                       may also belong to more than one process, so would have to sort that out.  Current
    #                       implementation doesn't have to worry about that.

    objective_mechanism = None

    # if activation_mech has outgoing projections
    if (lc.activation_mech_output.sendsToProjections and
            # if the ProcessingMechanisms to which activation_mech projects do not belong to any of the same processes
            # to which the mechanisms that project to activation_mech belong, then it should be treated as a TERMINAL
            # and is_target should be set to True
            not any(
                        # MODIFIED 4/5/17 OLD:
                        isinstance(projection.receiver.owner, ProcessingMechanism_Base) and

                        # # MODIFIED 4/5/17 NEW:
                        # # if it projects to an ObjectiveMechanism that is NOT used for learning,
                        # # then activation_mech should not be considered TERMINAL (and is_target should not be True)
                        # (isinstance(projection.receiver.owner, ProcessingMechanism_Base)
                        #             and not (isinstance(projection.receiver.owner, ObjectiveMechanism) and
                        #                                 projection.receiver.owner.role == LEARNING)) and
                        # # MODIFIED 4/5/17 END

                        any(                   # processes of ProcessingMechanisms to which activation_mech projects
                                    process in projection.receiver.owner.processes
                                                   # processes of mechanism that project to activation_mech
                                    for process in lc.activation_mech_input.owner.processes)
                        for projection in lc.activation_mech_output.sendsToProjections)):
        is_target = True

    else:

        for projection in lc.activation_mech_output.sendsToProjections:

            receiver_mech = projection.receiver.owner
            receiver_state = projection.receiver

            # Check if projection already projects to a LearningMechanism or an ObjectiveMechanism

            # lc.activation_mech projects to a LearningMechanism
            if isinstance(receiver_mech, LearningMechanism):

                # IMPLEMENTATION NOTE:  THIS IS A SANITY CHECK;  IF THE learning_projection ALREADY HAS A SENDER
                #                       THAT IS A LearningMechanism, THIS FUNCTION SHOULD NOT HAVE BEEN CALLED
                # If receiver_mech is a LearningMechanism that is the sender for the learning_projection,
                #    raise an exception since this function should not have been called.
                if learning_projection.sender is receiver_mech:
                    raise LearningAuxilliaryError("PROGRAM ERROR: "
                                                  "{} already has a LearningMechanism as its sender ({})".
                                                  format(learning_projection.name, receiver_mech.name))

                # If receiver_mech is a LearningMechanism that receives projections to its:
                #     - inputState[ACTIVATION_OUTPUT] from lc.activation_mech
                #     - inputState[ACTIVATION_INPUT] from lc.activation_mech_input.owner
                #         (i.e., the mechanism before lc.activation_mech in the learning sequence)
                #         then this should be the LearningMechanism for the learning_projection,
                #         so issue warning, assign it as the sender, and return
                if (receiver_state.name is ACTIVATION_OUTPUT and
                        any(projection.sender.owner is lc.activation_mech_input.owner
                            for projection in receiver_mech.input_states[ACTIVATION_INPUT].receivesFromProjections)):
                        warnings.warn("An existing LearningMechanism ({}) was found for and is being assigned to {}".
                                      format(receiver_mech.name, learning_projection.name))
                        learning_projection.sender = receiver_mech
                        return

            # lc.activation_mech already projects to an ObjectiveMechanism used for learning
            #    (presumably instantiated for another process);
            #    note:  doesn't matter if it is not being used for learning (then its just another ProcessingMechanism)
            elif isinstance(receiver_mech, ObjectiveMechanism) and LEARNING in receiver_mech._role:

                # ObjectiveMechanism is for learning but projection is not to its SAMPLE inputState
                if LEARNING in receiver_mech._role and not receiver_state.name is SAMPLE:
                    raise LearningAuxilliaryError("PROGRAM ERROR: {} projects to the {} rather than the {} "
                                                  "inputState of an ObjectiveMechanism for learning {}".
                                                  format(lc.activation_mech.name,
                                                         receiver_state.name,
                                                         SAMPLE,
                                                         receiver_mech.name))

                # IMPLEMENTATION NOTE:  THIS IS A SANITY CHECK;  IF THE learning_projection ALREADY HAS A SENDER
                #                       THAT IS A LearningMechanism, THIS FUNCTION SHOULD NOT HAVE BEEN CALLED
                # If the ObjectiveMechanism projects to a LearningMechanism that is the sender for the
                #     learning_projection, raise exception as this function should not have been called
                elif (isinstance(learning_projection.sender, LearningMechanism) and
                          any(learning_projection.sender.owner is project.receiver.owner
                              for projection in receiver_mech.outputState.sendsToProjections)):
                    raise LearningAuxilliaryError("PROGRAM ERROR:  {} already has an "
                                                  "ObjectiveMechanism ({}) and a "
                                                  "LearningMechanism ({}) assigned to it".
                                                  format(learning_projection.name,
                                                         receiver_mech.name,
                                                         learning_projection.sender.owner.name))

                else:
                    objective_mechanism = receiver_mech


        # FIX: CHECK THAT THE PROJECTION IS TO ANOTHER MECHANISM IN THE CURRENT PROCESS;
        #      OTHERWISE, ALLOW OBJECTIVE_MECHANISM TO BE IMPLEMENTED
        #      BUT HOW CAN IT KNOW THIS, SINCE UNTIL THE METHOD IS PART OF A COMPOSITION, IT DOESN'T KNOW WHAT
        #             COMPOSITION IS BEING CALLED.  -- LOOK AT LearningProjection_OLD TO SEE HOW IT WAS HANDLED THERE.
        #             OR IS A TARGET/OBJECTIVE MECHANISM ALWAYS ASSIGNED FOR A PROCESS, AND THEN DELETED IN SYSTEM GRAPH?
        #             IN THAT CASE, JUST IGNORE SECOND CONDITION BELOW (THAT IT HAS NO PROJECTIONS TO LEARNING_PROJECTIONS?

        # Next, determine whether an ObjectiveMechanism or LearningMechanism should be assigned as the sender
        # It SHOULD be an ObjectiveMechanism (i.e., TARGET) if either:
        #     - it has no outgoing projections or
        #     - it has no projections that receive a LearningProjection;
        #   in either case, lc.error_projection returns None.
        #   Note:  this assumes that LearningProjections are being assigned from the end of the pathway to the beginning.
        is_target = not lc.error_projection

    # INSTANTIATE learning function

    # Note: have to wait to do it here, as Backpropagation needs error_matrix,
    #       which depends on projection to ObjectiveMechanism

    # IMPLEMENTATION NOTE:
    #      THESE SHOULD BE MOVED (ALONG WITH THE SPECIFICATION FOR LEARNING) TO A DEDICATED LEARNING SPEC
    #      FOR PROCESS AND SYSTEM, RATHER THAN USING A LearningProjection
    # Get function used for learning and the learning_rate from their specification in the LearningProjection
    learning_function = learning_projection.learning_function
    learning_rate = learning_projection.learning_rate

    # REINFORCEMENT LEARNING FUNCTION
    if learning_function.componentName is RL_FUNCTION:

        activation_input = np.zeros_like(lc.activation_mech_input.value)
        activation_output = np.zeros_like(lc.activation_mech_output.value)

        # Force output activity and error arrays to be scalars
        error_output = error_signal  = np.array([0])
        learning_rate = learning_projection.learning_function.learning_rate

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     ACTIVATION FUNCTION AND/OR LEARNING RATE
        learning_function = Reinforcement(variable_default=[activation_input,activation_output,error_signal],
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
                                          format(learning_projection.name, BackPropagation.componentName))

        # Get error_mech values
        if is_target:
            error_output = np.ones_like(lc.activation_mech_output.value)
            error_signal = np.zeros_like(lc.activation_mech_output.value)
            error_matrix = np.identity(len(error_signal))
            # IMPLEMENTATION NOTE: Assign error_derivative to derivative of ProcessingInputState or SystemInputState
            #                      function when these are fully implemented as mechanisms
            # activation_derivative = Linear().derivative
            error_derivative = Linear().derivative

        else:
            error_output = np.zeros_like(lc.error_mech_output.value)
            error_signal = np.zeros_like(lc.error_signal_mech.output_states[ERROR_SIGNAL].value)
            error_matrix = lc.error_matrix
            try:
                error_derivative = lc.error_derivative
            except AttributeError:
                raise LearningAuxilliaryError("Function for error_mech of {} must have a derivative "
                                              "to be used with {}".
                                              format(learning_projection.name, BackPropagation.componentName))

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     DERIVATIVE, LEARNING_RATE, ERROR_MATRIX
        learning_function = BackPropagation(variable_default=[activation_input,
                                                              activation_output,
                                                              # error_output,
                                                              error_signal],
                                            activation_derivative_fct=activation_derivative,
                                            # # MODIFIED 3/5/17 OLD:
                                            # error_derivative_fct=error_derivative,
                                            # MODIFIED 3/5/17 NEW:
                                            error_derivative_fct=activation_derivative,
                                            # MODIFIED 3/5/17 END
                                            error_matrix=error_matrix,
                                            learning_rate=learning_rate,
                                            context=context)

    else:
        raise LearningAuxilliaryError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                  format(learning_function.componentName, learning_projection.name))


    # INSTANTIATE ObjectiveMechanism

    # If it is a TARGET, instantiate an ObjectiveMechanism and assign as learning_projection's sender
    if is_target:

        if objective_mechanism is None:
            # Instantiate ObjectiveMechanism
            # Notes:
            # * MappingProjections for ObjectiveMechanism's input_states will be assigned in its own call to Composition
            # * Need to specify both default_input_value and monitored_values since they may not be the same
            #    sizes (e.g., for RL the monitored_value for the sample may be a vector, but its input_value must be scalar)
            # SAMPLE inputState for ObjectiveMechanism should come from activation_mech_output
            # TARGET inputState for ObjectiveMechanism should be specified by string (TARGET),
            #     so that it is left free to later be assigned a projection from ProcessInputState and/or SystemInputState
            # Assign derivative of Linear to lc.error_derivative (as default, until TARGET projection is assigned);
            #    this will induce a simple subtraction of target-sample (i.e., implement a comparator)

            sample_input = target_input = error_output
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
                                                     function=LinearCombination(weights=[[-1], [1]]),
                                                     params=object_mech_params,
                                                     name=lc.activation_mech.name + " " + OBJECTIVE_MECHANISM,
                                                     context=context)

            objective_mechanism._role = LEARNING
            objective_mechanism._learning_role = TARGET

        try:
            lc.error_projection = objective_mechanism.input_state.receivesFromProjections[0]
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

        # INSTANTIATE LearningMechanism

    # - LearningMechanism incoming projections (by inputState):
    #    ACTIVATION_INPUT:
    #        MappingProjection from activation_mech_input
    #    ACTIVATION_OUTPUT:
    #        MappingProjection from activation_mech_output
    #    ERROR_SIGNAL:
    #        specified in error_source argument:
    #        Note:  the error_source for LearningMechanism is set in lc.error_signal_mech:
    #            if is_target, this comes from the primary outputState of objective_mechanism;
    #            otherwise, it comes from outputStates[ERROR_SIGNAL] of the LearningMechanism for lc.error_mech
    # - Use of AUTO_ASSIGN_MATRIX for the MappingProjections is safe, as compatibility of senders and receivers
    #    is checked in the instantiation of the learning_function

    error_source = lc.error_signal_mech

    learning_mechanism = LearningMechanism(variable=[activation_input,
                                                     activation_output,
                                                     error_signal],
                                           error_source=error_source,
                                           function=learning_function,
                                           name = lc.activation_mech_projection.name + " " +LEARNING_MECHANISM,
                                           context=context)

    # IMPLEMENTATION NOTE:
    # ADD ARGUMENTS TO LearningMechanism FOR activation_input AND activation_output, AND THEN INSTANTIATE THE
    # MappingProjections BELOW IN CALL TO HELPER METHODS FROM LearningMechanism._instantiate_attributes_before_function
    # (FOLLOWING DESIGN OF _instantiate_error_signal_projection IN LearningMechanism):

    # Assign MappingProjection from activation_mech_input to LearningMechanism's ACTIVATION_INPUT inputState
    MappingProjection(sender=lc.activation_mech_input,
                      receiver=learning_mechanism.input_states[ACTIVATION_INPUT],
                      matrix=IDENTITY_MATRIX,
                      name = lc.activation_mech_input.owner.name + ' to ' + ACTIVATION_INPUT,
                      context=context)

    # Assign MappingProjection from activation_mech_output to LearningMechanism's ACTIVATION_OUTPUT inputState
    MappingProjection(sender=lc.activation_mech_output,
                      receiver=learning_mechanism.input_states[ACTIVATION_OUTPUT],
                      matrix=IDENTITY_MATRIX,
                      name = lc.activation_mech_output.owner.name + ' to ' + ACTIVATION_OUTPUT,
                      context=context)

    # Assign learning_mechanism as sender of learning_projection and return
    # Note: learning_projection still has to be assigned to the learning_mechanism's outputState;
    #       however, this requires that it's variable be assigned (which occurs in the rest of its
    #       _instantiate_sender method, from which this was called) and that its value be assigned
    #       (which occurs in its _instantiate_function method).
    learning_projection.sender = learning_mechanism.outputState


class LearningComponents(object):
    """Gets components required to instantiate LearningMechanism and its Objective Function for a LearningProjection

    Has attributes for the following learning components relevant to a `LearningProjection`,
    each of which is found and/or validated if necessary before assignment:

    * `activation_mech_projection` (`MappingProjection`):  one being learned)
    * `activation_mech_input` (`OutputState`):  input to mechanism to which projection being learned projections
    * `activation_mech` (`ProcessingMechanism`):  mechanism to which projection being learned projects
    * `activation_mech_output` (`OutputState`):  output of activation_mech
    * `activation_mech_fct` (function):  function of mechanism to which projection being learned projects
    * `activation_derivative` (function):  derivative of function for activation_mech
    * `error_projection` (`MappingProjection`):  next projection in learning sequence after activation_mech_projection
    * `error_matrix` (`ParameterState`):  parameterState of error_projection with error_matrix
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

    IMPLEMENTATION NOTE:  The helper methods in this class (that assign values to the various attributes)
                          respect membership in a process;  that is, they do not assign attribute values to
                          objects that belong to a process in which the root attribute (self.activation_mech)
                          belongs
    """

    def __init__(self, learning_projection, context=None):

        self._validate_learning_projection(learning_projection)
        self.learning_projection = learning_projection

        self._activation_mech_projection = None
        self._activation_mech = None
        self._activation_mech_input = None
        self._activation_mech_fct = None
        self._activation_derivative = None
        self._activation_mech_output = None
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
        self.activation_mech
        self.activation_mech_input
        self.activation_mech_fct
        self.activation_derivative
        self.activation_mech_output
        self.error_projection
        self.error_matrix
        self.error_mech
        self.error_derivative
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
    # activation_mech_output:  output of activation_mech (OutputState)
    @property
    def activation_mech_output(self):
        def _get_act_sample():
            if not self.activation_mech:
                return None
            # If MONITOR_FOR_LEARNING specifies an outputState, use that
            try:
                sample_state_name = self.activation_mech.paramsCurrent[MONITOR_FOR_LEARNING]
                self.activation_mech_output = self.activation_mech.output_states[sample_state_name]
                if not isinstance(self.activation_mech_output, OutputState):
                    raise LearningAuxilliaryError("The specification of the MONITOR_FOR_LEARNING parameter ({}) "
                                                  "for {} is not an outputState".
                                                  format(self.activation_mech_output, self.learning_projection))
            except KeyError:
                # No outputState specified so use primary outputState
                try:
                    self.activation_mech_output = self.activation_mech.outputState
                except AttributeError:
                    raise LearningAuxilliaryError("activation_mech_output not identified: activation_mech ({})"
                                                  "not appear to have been assigned a primary outputState.".
                                                  format(self.learning_projection))
            return self.activation_mech.outputState

        return self._activation_mech_output or _get_act_sample()

    @activation_mech_output.setter
    def activation_mech_output(self, assignment):
        if isinstance(assignment, (OutputState)):
            self._activation_mech_output =assignment
        else:
            raise LearningAuxilliaryError("PROGRAM ERROR: illegal assignment to activation_mech_output; "
                                          "it must be a OutputState.")


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
    # error_projection:  one that has the error_matrix (MappingProjection)
    @property
    def error_projection(self):
        # Find from activation_mech:
        def _get_error_proj():
            if not self.activation_mech_output:
                return None
            projections = self.activation_mech_output.sendsToProjections
            # MODIFIED 3/11/17 OLD:
            # error_proj must be a MappingProjection that has a LearningProjection to it
            error_proj = next((projection for projection in projections if
                              (isinstance(projection, MappingProjection) and projection.has_learning_projection)),None)
            # # MODIFIED 3/11/17 NEW:
            # # error_proj must be a MappingProjection:
            # #   that project to another mechanism in the same process and
            # #   that has a LearningProjection
            # error_proj = next((projection for projection in projections if
            #                    (isinstance(projection, MappingProjection) and
            #                     projection.has_learning_projection and
            #                     any(process in projection.receiver.owner.processes
            #                         for process in self.activation_mech.processes))),
            #                   None)
            # MODIFIED 3/11/17 END
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
    # error_derivative:  derivative of function of error_mech (function)
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
        def _get_error_signal_mech():
            if not self.error_matrix:
                return None
            # search the projections to the error_matrix parameter state for a LearningProjection
            learning_proj = next((proj for proj in self.error_matrix.receivesFromProjections
                                 if isinstance(proj, LearningProjection)), None)
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
            # MODIFIED 3/11/17 NEW:
            # If the learning_mech found has already been assigned to a process that not a process to which the
            #    preceding mechanism in the sequence (self.activation_mech_input.owner) belongs, then the current
            #    mechanism (self.activation_mech) is the TERMINAL for its process, so look for an ObjectiveMechanism
            if learning_mech.processes and not any(process in self.activation_mech_input.owner.processes
                                                   for process in learning_mech.processes):
                new_learning_proj = next((projection for projection in
                                          self.activation_mech.outputState.sendsToProjections
                                          if isinstance(projection.receiver.owner, ObjectiveMechanism)), None)
                if new_learning_proj is None:
                    return None
                else:
                    learning_mech = new_learning_proj.receiver.owner
            # MODIFIED 3/11/17 END

            self.error_signal_mech = learning_mech
            return self.error_signal_mech

        return self._error_signal_mech or _get_error_signal_mech()

    @error_signal_mech.setter
    def error_signal_mech(self, assignment):
        if (assignment is None or
                isinstance(assignment, LearningMechanism) or
                (isinstance(assignment, ObjectiveMechanism) and assignment._role is LEARNING)):
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
                    self.error_signal_mech_output = self.error_signal_mech.output_states[ERROR_SIGNAL]
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
    #                                    for proj in learning_mech.input_states[ERROR_SIGNAL].receivesFromProjections
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
